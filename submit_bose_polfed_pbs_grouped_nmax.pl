#!/usr/bin/env perl
#
# Prepare one PBS job per (U,W) pair for the single-species Bose-Hubbard chain.
# Each job runs one Julia process, and that process handles nreal disorder
# realizations with checkpointed output.
#
# Typical cluster use in the new cluster directory:
#   mkdir -p /home/$USER/POLFED_bosons
#   cd /home/$USER/POLFED_bosons
#   perl submit_bose_polfed_pbs_grouped_nmax.pl \
#     --workdir /home/$USER/POLFED_bosons \
#     --module julia/1.10 \
#     --L 10 --N 10 --nmax 2 \
#     --U-list 4.0 --W-list 4.0,6.0,8.0 --submit

use strict;
use warnings;
use Getopt::Long;
use File::Path qw(make_path);
use Cwd qw(getcwd);
use File::Spec;

my $julia_script  = "run_bose_hubbard_polfed_N_fixed_nmax.jl";
my $julia_cmd     = "julia";
my $julia_project = ".";
my @modules       = ();

my $U_list        = "4.0";
my $W_list        = "6.0";

my $L             = 10;
my $N             = -1;
my $nmax          = 2;
my $t             = 1.0;

my $nreal         = 100;
my $seed0         = 1234;
my $howmany       = 500;
my $target        = "middle";
my $block         = 4;
my $boundary      = "open";

my $outdir        = "pbs_bose_polfed_grouped";
my $workdir       = "";
my $walltime      = "24:00:00";
my $mem           = "8gb";
my $nodes         = 1;
my $ppn           = 1;
my $queue         = "";
my $job_prefix    = "bosepolfed";

my $submit        = 0;
my $save_fields   = 0;
my $full          = 0;

GetOptions(
    "julia-script=s"   => \$julia_script,
    "julia-cmd=s"      => \$julia_cmd,
    "julia-project=s"  => \$julia_project,
    "module=s@"        => \@modules,
    "U-list=s"         => \$U_list,
    "W-list=s"         => \$W_list,
    "L=i"              => \$L,
    "N=i"              => \$N,
    "nmax=i"           => \$nmax,
    "t=f"              => \$t,
    "nreal=i"          => \$nreal,
    "seed0=i"          => \$seed0,
    "howmany=i"        => \$howmany,
    "target=s"         => \$target,
    "block=i"          => \$block,
    "boundary=s"       => \$boundary,
    "outdir=s"         => \$outdir,
    "workdir=s"        => \$workdir,
    "walltime=s"       => \$walltime,
    "mem=s"            => \$mem,
    "nodes=i"          => \$nodes,
    "ppn=i"            => \$ppn,
    "queue=s"          => \$queue,
    "job-prefix=s"     => \$job_prefix,
    "submit!"          => \$submit,
    "save-fields!"     => \$save_fields,
    "full!"            => \$full,
) or die "Error in command line arguments\n";

sub parse_list {
    my ($s) = @_;
    $s =~ s/\s+//g;
    die "Empty parameter list\n" if $s eq "";
    my @x = split(/,/, $s);
    for my $v (@x) {
        die "Bad numeric value in list: $v\n"
            unless $v =~ /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$/;
    }
    return @x;
}

sub safe_float_string {
    my ($x) = @_;
    my $s = "$x";
    $s =~ s/\+//g;
    $s =~ s/-/m/g;
    $s =~ s/\./p/g;
    $s =~ s/[eE]/e/g;
    return $s;
}

sub shell_quote {
    my ($s) = @_;
    my $q = "$s";
    $q =~ s/'/'"'"'/g;
    return "'$q'";
}

sub seed_from_parameters {
    my ($seed0, $U, $W, $L, $N_eff, $nmax) = @_;

    my $u_int = int(1000.0 * $U + ($U >= 0 ? 0.5 : -0.5));
    my $w_int = int(1000.0 * $W + ($W >= 0 ? 0.5 : -0.5));

    my $seed = $seed0
             + 1000003 * $u_int
             + 10007   * $w_int
             + 101     * $L
             + 17      * $N_eff
             + 19      * $nmax;

    $seed = $seed % 2147483647;
    $seed = 12345 if $seed <= 0;
    return $seed;
}

sub validate_simple_token {
    my ($name, $value) = @_;
    die "$name cannot be empty\n" if !defined($value) || $value eq "";
    die "$name contains unsafe characters: $value\n"
        if $value =~ /[\r\n]/;
}

validate_simple_token("julia-cmd", $julia_cmd);
validate_simple_token("target", $target);
validate_simple_token("boundary", $boundary);

die "L must be positive\n" unless $L > 0;
my $N_eff = $N < 0 ? $L : $N;
die "N must be nonnegative after defaulting. Got $N_eff\n"
    unless $N_eff >= 0;
die "nmax must be positive\n" unless $nmax > 0;
die "N must satisfy N <= L*nmax. Got N=$N_eff, L=$L, nmax=$nmax\n"
    unless $N_eff <= $L * $nmax;
die "nreal must be positive\n" unless $nreal > 0;
die "howmany must be positive\n" unless $howmany > 0;
die "block must be positive\n" unless $block > 0;
die "ppn must be positive\n" unless $ppn > 0;

my @Us = parse_list($U_list);
my @Ws = parse_list($W_list);

my $cwd = getcwd();
$workdir = $cwd if $workdir eq "";

if ($workdir =~ /^[A-Za-z]:\\/ || $workdir =~ /\\/) {
    warn "WARNING: workdir looks like a Windows path: $workdir\n";
    warn "         PBS compute nodes usually need a Linux path such as /home/user/POLFED_bosons.\n";
}

my $local_script_path = $julia_script;
$local_script_path = File::Spec->catfile($cwd, $julia_script)
    if !File::Spec->file_name_is_absolute($local_script_path);
die "Julia script not found locally: $local_script_path\n"
    unless -e $local_script_path;

make_path($outdir);
make_path(File::Spec->catdir($outdir, "jobs"));
make_path(File::Spec->catdir($outdir, "logs"));
make_path(File::Spec->catdir($outdir, "data"));

my $n_jobs = 0;

foreach my $U (@Us) {
    foreach my $W (@Ws) {
        my $seed = seed_from_parameters($seed0, $U, $W, $L, $N_eff, $nmax);

        my $Utag = safe_float_string($U);
        my $Wtag = safe_float_string($W);
        my $sector = "L_${L}/N_${N_eff}_nmax_${nmax}";
        my $data_dir = "$workdir/$outdir/data/U_${Utag}/W_${Wtag}/$sector";

        my $jobtag = sprintf("U%s_W%s_L%d_N%d_nm%d", $Utag, $Wtag, $L, $N_eff, $nmax);
        my $jobname = sprintf("%s_%s", $job_prefix, $jobtag);
        $jobname = substr($jobname, 0, 30);

        my $jobfile = File::Spec->catfile($outdir, "jobs", "job_${jobtag}.pbs");
        my $stdout = "$workdir/$outdir/logs/${jobtag}.out";
        my $stderr = "$workdir/$outdir/logs/${jobtag}.err";

        open(my $fh, ">", $jobfile) or die "Cannot write $jobfile: $!\n";

        print $fh "#!/bin/bash\n";
        print $fh "#PBS -N $jobname\n";
        print $fh "#PBS -l walltime=$walltime\n";
        print $fh "#PBS -l mem=$mem\n";
        print $fh "#PBS -l nodes=$nodes:ppn=$ppn\n";
        print $fh "#PBS -o $stdout\n";
        print $fh "#PBS -e $stderr\n";
        print $fh "#PBS -V\n";
        print $fh "#PBS -q $queue\n" if $queue ne "";
        print $fh "\n";
        print $fh "set -euo pipefail\n";
        print $fh "echo \"Job started on \$(hostname) at \$(date)\"\n";
        print $fh "echo \"PBS_JOBID=\${PBS_JOBID:-unknown}\"\n";
        print $fh "echo \"PBS_O_WORKDIR=\${PBS_O_WORKDIR:-unknown}\"\n";
        print $fh "cd " . shell_quote($workdir) . "\n";
        print $fh "\n";

        if (@modules) {
            print $fh "module purge\n";
            for my $module (@modules) {
                validate_simple_token("module", $module);
                print $fh "module load " . shell_quote($module) . "\n";
            }
            print $fh "\n";
        }

        print $fh "export OMP_NUM_THREADS=$ppn\n";
        print $fh "export OPENBLAS_NUM_THREADS=$ppn\n";
        print $fh "export JULIA_NUM_THREADS=$ppn\n";
        print $fh "mkdir -p " . shell_quote($data_dir) . "\n";
        print $fh "\n";

        my @cmd_parts = (
            $julia_cmd,
            "--project=" . shell_quote($julia_project),
            shell_quote($julia_script),
            "--L", $L,
            "--N", $N_eff,
            "--nmax", $nmax,
            "--t", $t,
            "--U", $U,
            "--W", $W,
            "--nreal", $nreal,
            "--seed", $seed,
            "--howmany", $howmany,
            "--target", shell_quote($target),
            "--block", $block,
            "--boundary", shell_quote($boundary),
            "--outdir", shell_quote($data_dir),
        );

        push @cmd_parts, "--save-fields" if $save_fields;
        push @cmd_parts, "--full" if $full;

        print $fh join(" ", @cmd_parts) . "\n";
        print $fh "echo \"Job finished at \$(date)\"\n";
        close($fh);

        $n_jobs++;

        if ($submit) {
            system("qsub", $jobfile) == 0 or die "qsub failed for $jobfile\n";
        } else {
            print "[dry-run] prepared $jobfile\n";
        }
    }
}

print "\nPrepared jobs: $n_jobs\n";
print "Parameter combinations: ", scalar(@Us), " U values x ", scalar(@Ws), " W values\n";
print "Realizations per job: $nreal\n";
print "Effective N: $N_eff\n";
print "nmax: $nmax\n";
print "Workdir used inside PBS scripts: $workdir\n";

if ($submit) {
    print "Submitted jobs to PBS.\n";
} else {
    print "Dry run only. Add --submit to call qsub.\n";
}
