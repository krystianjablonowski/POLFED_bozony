#!/usr/bin/env perl
#
# Generic PBS submitter for full-diagonalization bosonic filling observables.
# It creates one job per (N, nmax, U, W), using the bosonic observable runner
# from Wstar_filling_L8_bosons_study.  The runner itself accepts arbitrary L;
# only the old L8 submitter was hard-coded to L=8.
#
# Example:
#   perl submit_boson_filling_observables_pbs.pl \
#     --workdir /home/2/kj405942/POLFED_bosons \
#     --L 7 \
#     --sectors 7:2,7:3,7:4,7:5 \
#     --base-outdir cluster_L7_boson_entropy_observables \
#     --submit

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use File::Path qw(make_path);
use File::Spec;
use Cwd qw(abs_path);

my $workdir = '';
my $julia_cmd = 'julia';
my $runner = '';
my $L = 7;
my $sectors = '7:2,7:3,7:4,7:5';
my $U_list = '1,2,3,4,5,6,7,8,9,10';
my $W_list = '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15';
my $nreal = 300;
my $middle_count = 0;
my $middle_fraction = 0.5;
my $walltime = '24:00:00';
my $mem = '24gb';
my $ppn = 1;
my $boundary = 'periodic';
my $max_dim = 12000;
my $base_outdir = 'cluster_boson_entropy_observables';
my $submit = 0;
my $compute_entanglement = 0;

GetOptions(
    'workdir=s' => \$workdir,
    'julia-cmd=s' => \$julia_cmd,
    'runner=s' => \$runner,
    'L=i' => \$L,
    'sectors=s' => \$sectors,
    'U-list=s' => \$U_list,
    'W-list=s' => \$W_list,
    'nreal=i' => \$nreal,
    'middle-count=i' => \$middle_count,
    'middle-fraction=f' => \$middle_fraction,
    'walltime=s' => \$walltime,
    'mem=s' => \$mem,
    'ppn=i' => \$ppn,
    'boundary=s' => \$boundary,
    'max-dim=i' => \$max_dim,
    'base-outdir=s' => \$base_outdir,
    'compute-entanglement!' => \$compute_entanglement,
    'submit!' => \$submit,
) or die "Error in command line arguments\n";

die "--workdir is required\n" unless $workdir;
die "--L must be positive\n" unless $L > 0;
die "--nreal must be positive\n" unless $nreal > 0;
die "--max-dim must be positive\n" unless $max_dim > 0;
die "--ppn must be positive\n" unless $ppn > 0;

$workdir = abs_path($workdir) // die "Cannot resolve workdir: $workdir\n";
if ($runner eq '') {
    $runner = File::Spec->catfile($workdir, 'Wstar_filling_L8_bosons_study', 'run_l8_boson_filling_observables.jl');
}
die "Runner not found: $runner\n" unless -f $runner;

my $root = File::Spec->catdir($workdir, $base_outdir);
make_path($root);

sub parse_list {
    my ($text) = @_;
    $text =~ s/\s+//g;
    return grep { length($_) } split /,/, $text;
}

sub tag {
    my ($value) = @_;
    $value = "$value";
    $value =~ s/-/m/g;
    $value =~ s/\./p/g;
    return $value;
}

sub shell_quote {
    my ($value) = @_;
    $value =~ s/'/'"'"'/g;
    return "'$value'";
}

my @U_values = parse_list($U_list);
my @W_values = parse_list($W_list);
my @sector_values = parse_list($sectors);
my $prepared = 0;

for my $sector (@sector_values) {
    my ($N, $nmax) = split /:/, $sector;
    die "Bad sector '$sector'; expected N:nmax\n" unless defined $N && defined $nmax;
    die "Bad sector '$sector'; require N >= 0 and nmax > 0\n" unless $N >= 0 && $nmax > 0;
    die "Bad sector '$sector'; require N <= L*nmax\n" unless $N <= $L * $nmax;

    my $nmax_root = File::Spec->catdir($root, "nmax_$nmax");
    my $jobs_dir = File::Spec->catdir($nmax_root, 'jobs');
    my $logs_dir = File::Spec->catdir($nmax_root, 'logs');
    my $data_dir = File::Spec->catdir($nmax_root, 'data');
    make_path($jobs_dir, $logs_dir, $data_dir);

    for my $U (@U_values) {
        for my $W (@W_values) {
            my $name = sprintf('bobs_L%d_N%d_nm%d_U%s_W%s', $L, $N, $nmax, tag($U), tag($W));
            my $job_path = File::Spec->catfile($jobs_dir, "$name.pbs");
            my $stdout = File::Spec->catfile($logs_dir, "$name.out");
            my $stderr = File::Spec->catfile($logs_dir, "$name.err");
            my $outdir = File::Spec->catdir(
                $data_dir, "L$L", "N_${N}_nmax_${nmax}", "U_" . tag($U), "W_" . tag($W)
            );
            make_path($outdir);
            my $seed = 3000001 + 1000003 * $L + 10007 * $N + 1009 * $nmax + 101 * int($U * 10) + int($W * 10);
            my $entanglement_flag = $compute_entanglement ? ' --compute-entanglement' : '';

            open my $fh, '>', $job_path or die "Cannot write $job_path: $!\n";
            print {$fh} "#!/bin/bash\n";
            print {$fh} "#PBS -N $name\n";
            print {$fh} "#PBS -l walltime=$walltime\n";
            print {$fh} "#PBS -l mem=$mem\n";
            print {$fh} "#PBS -l nodes=1:ppn=$ppn\n";
            print {$fh} "#PBS -o $stdout\n";
            print {$fh} "#PBS -e $stderr\n";
            print {$fh} "#PBS -V\n\n";
            print {$fh} "set -euo pipefail\n";
            print {$fh} "cd ", shell_quote($workdir), "\n";
            print {$fh} "export OMP_NUM_THREADS=$ppn\n";
            print {$fh} "export OPENBLAS_NUM_THREADS=$ppn\n";
            print {$fh} "export JULIA_NUM_THREADS=$ppn\n";
            print {$fh} "echo \"Started \$(date) on \$(hostname)\"\n";
            print {$fh} shell_quote($julia_cmd), " --project=. ", shell_quote($runner),
                " --L $L --N $N --nmax $nmax --t 1",
                " --U $U --W $W --nreal $nreal --middle-count $middle_count",
                " --middle-fraction $middle_fraction",
                " --seed $seed --boundary ", shell_quote($boundary),
                " --max-dim $max_dim",
                " --outdir ", shell_quote($outdir), $entanglement_flag, "\n";
            print {$fh} "echo \"Finished \$(date)\"\n";
            close $fh;
            chmod 0755, $job_path;
            $prepared++;

            if ($submit) {
                system('qsub', $job_path) == 0 or warn "qsub failed for $job_path\n";
            } else {
                print "[dry-run] $job_path\n";
            }
        }
    }
}

print "\nPrepared jobs: $prepared\n";
print "L: $L\n";
print "Sectors N:nmax: ", join(', ', @sector_values), "\n";
print "Results are separated below: $root/nmax_<value>/\n";
print $submit ? "Jobs submitted.\n" : "Dry run only. Add --submit to call qsub.\n";
