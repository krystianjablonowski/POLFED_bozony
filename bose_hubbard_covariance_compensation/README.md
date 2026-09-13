# Model kompensacji kowariancyjnej dla bozonów

Program sprawdza bezparametrową hipotezę

\[
U^*_{\mathrm{comp}}=-\frac{\operatorname{Cov}(E_{\mathrm{dis}},Q)}{\operatorname{Var}(Q)},
\qquad Q=\sum_i n_i(n_i-1),
\]

oraz jej wersję opartą na krawędziach grafu Focka. Nie dopasowuje do danych
ED żadnego mnożnika ani przesunięcia.

## Konwencja fizyczna

Kod używa dokładnie

\[
H=-t\sum_{i=1}^{L-1}(b_i^\dagger b_{i+1}+\mathrm{h.c.})
+\sum_i\epsilon_i n_i+U\sum_i n_i(n_i-1),
\qquad \epsilon_i\sim\mathcal U[-W,W].
\]

- brak czynnika `1/2` przy `U`;
- nieporządek jest z `-W` do `W`, a nie z `-W/2` do `W/2`;
- otwarte warunki brzegowe;
- seedy `SeedSequence([20260907, L, sample_id])`, zgodne z obliczeniami ED;
- wybieranych jest 20 stanów najbliższych gęstości energii `0.5`;
- stany są liczone tylko dla `U=0`.

Dla sektorów używanych tutaj (`nmax >= N`) stany wielu ciał przy `U=0` są
konstruowane dokładnie z orbitali jednocząstkowego modelu Andersona. Nie ma
pełnej diagonalizacji Hamiltonianu Bosego–Hubbarda dla kolejnych wartości
`U`. Kod mierzy residuum każdego skonstruowanego stanu i przerywa zadanie,
jeżeli przekracza ono tolerancję.

## Pliki

- `covariance_core.py` — baza Focka, hopping, stany `U=0` i oba predyktory;
- `covariance_compensation.py` — zadania, agregacja, bootstrap, CSV i wykresy;
- `config_L4_pilot.json` — pierwszy wymagany test `L=N=nmax=4`;
- `config_L7_production.json` — sektory `L=7`, `N=nmax=4,5,6,7`;
- `pbs_worker.sh`, `submit_pbs.sh` — obliczenia PBS;
- `pbs_analyze.sh`, `submit_analysis.sh` — agregacja i wykresy PBS;
- `tests/` — testy poprawności implementacji.

## 1. Przesłanie na Kruka

Poniższą komendę uruchom w **Windows CMD**, w jednym wierszu (bez znaku
kontynuacji PowerShell `` ` ``):

```bat
cd "C:\Users\avoga\OneDrive\Dokumenty\POLFED"
scp -r bose_hubbard_covariance_compensation kj405942@kruk-host.fuw.edu.pl:/home/2/kj405942/POLFED_bosons/
```

## 2. Testy na Kruku

```bash
ssh kj405942@kruk-host.fuw.edu.pl
conda activate /home/2/kj405942/conda_envs/boson_peak_analysis
cd /home/2/kj405942/POLFED_bosons/bose_hubbard_covariance_compensation
chmod u+x pbs_worker.sh pbs_analyze.sh submit_pbs.sh submit_analysis.sh
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
python covariance_compensation.py validate --config config_L4_pilot.json
```

Testy muszą zakończyć się komunikatem `OK`. Walidacja pilota powinna podać
`L=4 N=4 nmax=4`, wymiar 35 oraz dokładną konstrukcję z orbitali Andersona.

## 3. Najpierw pilot L=4

```bash
bash submit_pbs.sh config_L4_pilot.json
```

Skrypt próbuje wysłać tablicę PBS bez limitu jednoczesności. Jeżeli dana
wersja PBS nie obsługuje tablic, automatycznie wysyła wszystkie zwykłe
zadania. Sprawdzanie:

```bash
qstat -u kj405942
python covariance_compensation.py status --config config_L4_pilot.json
grep -R -n -E 'Traceback|Error|Killed|MemoryError|walltime' results_L4_pilot/logs || true
```

Po otrzymaniu `Complete 25/25; missing 0` uruchom analizę:

```bash
bash submit_analysis.sh config_L4_pilot.json
```

Kontrola analizy:

```bash
tail -n 100 results_L4_pilot/logs/analyze.out
cat results_L4_pilot/analysis/summary.txt
find results_L4_pilot/analysis/figures -name '*.png' | wc -l
find results_L4_pilot/analysis/figures -name '*.pdf' | wc -l
```

## 4. Obliczenia L=7 i porównanie z ED

Utwórz katalog z małymi plikami porównawczymi:

```bash
mkdir -p comparison_data
```

Umieść w nim:

- `comparison_data/entropy_peak_points.csv` — maksima entropii ED;
- `comparison_data/peaks_L7_corrected.csv` — wyniki modelu funkcji mieszania,
  jeżeli mają być pokazane jako `U_M*`.

Najprościej przesłać je z **Windows CMD** (nie z sesji SSH na Kruku):

```bat
scp "C:\Users\avoga\Downloads\entropy_peak_points.csv" kj405942@kruk-host.fuw.edu.pl:/home/2/kj405942/POLFED_bosons/bose_hubbard_covariance_compensation/comparison_data/
scp "C:\Users\avoga\Downloads\peaks_L7_corrected.csv" kj405942@kruk-host.fuw.edu.pl:/home/2/kj405942/POLFED_bosons/bose_hubbard_covariance_compensation/comparison_data/
```

Następnie:

```bash
bash submit_pbs.sh config_L7_production.json
qstat -u kj405942
python covariance_compensation.py status --config config_L7_production.json
```

Konfiguracja produkcyjna tworzy 100 zadań: 4 sektory × 25 wartości `W`.
Każde zadanie liczy 300 realizacji zgodnych seed po seedzie
z istniejącymi obliczeniami ED. Nie ma ograniczenia liczby jednocześnie działających
zadań. Stan jest atomowo zapisywany co 10 realizacji, więc przerwane zadanie
można bezpiecznie wznowić. Po otrzymaniu `Complete 100/100; missing 0`:

```bash
bash submit_analysis.sh config_L7_production.json
qstat -u kj405942
```

Alternatywnie pliki porównawcze można podać jawnie:

```bash
bash submit_analysis.sh config_L7_production.json comparison_data/entropy_peak_points.csv comparison_data/peaks_L7_corrected.csv
```

## 5. Wyniki

W `results_L7_production/analysis/` powstaną:

- `predictor_points.csv` — `U_mean`, główny `U_pool`, `U_edge_mean` i
  `U_edge_pool` wraz z 95% CI;
- `linear_fits.csv` — nachylenia, wyrazy wolne, 95% CI, `R²` i RMSE;
- `comparison_metrics.csv` — RMSE, MAE i obciążenie względem maksimów entropii;
  zakres `common_with_M` pozwala uczciwie porównać predyktor z modelem
  funkcji mieszania na dokładnie tych samych punktach `W`;
- `state_records.csv.gz` — surowe wyniki stan po stanie;
- `summary.txt`;
- `figures/*.png` i `figures/*.pdf`.

Parowany bootstrap losuje te same identyfikatory realizacji dla wszystkich
wartości `W` i wszystkich predyktorów. CSV raportuje również udział ujemnej
kowariancji oraz dodatnich i ujemnych wartości `U_comp`. Dopasowania liniowe
są wykonywane na wspólnym zakresie `W/t=0.8..2.5`; obliczenia i wykresy
obejmują pełny zakres `0.1..2.5`. Niepewność dopasowania ED i wcześniejszego
modelu jest propagowana z przedziałów zapisanych w ich plikach CSV.

## 6. Wysłanie kodu i niewielkich wyników na GitHub

Surowe pliki zadań i duży `state_records.csv.gz` są celowo ignorowane przez
Git. Po wykonaniu analizy na Kruku:

```bash
cd /home/2/kj405942/POLFED_bosons
git status --short
git add bose_hubbard_covariance_compensation/*.py \
        bose_hubbard_covariance_compensation/*.json \
        bose_hubbard_covariance_compensation/*.sh \
        bose_hubbard_covariance_compensation/*.md \
        bose_hubbard_covariance_compensation/requirements.txt \
        bose_hubbard_covariance_compensation/.gitignore \
        bose_hubbard_covariance_compensation/tests \
        bose_hubbard_covariance_compensation/comparison_data \
        bose_hubbard_covariance_compensation/results_L4_pilot/analysis \
        bose_hubbard_covariance_compensation/results_L7_production/analysis
git status --short
git commit -m "Add covariance compensation test for bosonic entropy peaks"
git push origin HEAD
```

Jeżeli nie wykonano jeszcze jednego z przebiegów, usuń jego ścieżkę z komendy
`git add`. `.gitignore` zapobiegnie przypadkowemu dodaniu dużych danych
surowych, logów i plików zadań.

## Ważna interpretacja

Program nie wymusza dodatniego wyniku. Jeżeli kowariancja będzie dodatnia,
`U_comp` pozostanie ujemne i zostanie zapisane. Jeżeli `U_pool(W)` nie pokryje
się z `U_S(W)` w granicach niepewności, jest to negatywny wynik testu hipotezy,
a nie powód do skalowania predyktora współczynnikiem dopasowanym do ED.
