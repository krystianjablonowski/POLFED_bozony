# Poprawiony model gwiazdy Bosego–Hubbarda dla L=7

Wersja 3.0 oblicza małe lokalne gwiazdy (maksymalnie `13 x 13`), a nie pełną
diagonalizację modelu wielu ciał. Wyniki starszych wersji nie są zgodne z tym
formatem i nie zostaną przez program wczytane.

Najważniejsze konwencje:

- otwarty łańcuch, `t=1` i `U*sum_i n_i(n_i-1)`;
- `epsilon_i/W` jest niezależne i jednostajne na `[-1,1]`;
- seed realizacji jest identyczny z ED:
  `SeedSequence([master_seed, L, sample_id])`;
- konfiguracje centralne to 20 stanów o energii diagonalnej najbliższej
  znormalizowanemu środkowi widma `0.5`, zgodnie z wyborem 20 stanów ED;
- `M_existing` zachowuje wcześniejszą definicję projektu (ważone kanały
  kompensujące `k>0` i `|epsilon_i-epsilon_j|`);
- `M_graph_sum`, `M_graph_per_edge`, `S2_graph_sum` i `S2_graph_per_edge`
  liczą bezpośrednio wszystkie dozwolone krawędzie grafu Focka;
- `Sstar` diagonalizuje gwiazdę złożoną z konfiguracji centralnej i wszystkich
  jej sąsiadów; remisy maksymalnego nakładu są uśredniane.

`config_L7_corrected.json` liczy `N=nmax=4,5,6,7`, `W/t=0.8,...,2.5`,
`U/t=0,...,0.8` z krokiem `0.005` i 300 tych samych realizacji nieporządku dla
każdego sektora i każdego `W`. Są 72 zadania PBS — po jednym na parę
`(N,W)` — bez ograniczenia liczby zadań uruchamianych równocześnie po stronie
skryptu. Scheduler Kruka decyduje, ile naprawdę uruchomi naraz.

## Test lokalny lub na Kruku

```bash
python -m unittest discover -s tests -v
python run_l7_star.py validate --config config_L7_corrected.json
```

Powinno przejść 15 testów. Testy obejmują niezależne zbudowanie małego pełnego
Hamiltonianu dla `L=N=4`, porównanie jego lokalnego bloku z macierzą gwiazdy,
zgodność seeda z ED, detuning, normalizację i niezmienniczość na permutację
bazy.

## Wysłanie katalogu z Windows

W CMD (jedna linia, bez znaku `` ` ``):

```text
cd C:\Users\avoga\OneDrive\Dokumenty\POLFED
scp -r bose_hubbard_star_L7_fast kj405942@kruk-host.fuw.edu.pl:/home/2/kj405942/POLFED_bosons/
```

## Uruchomienie na Kruku

```bash
conda activate /home/2/kj405942/conda_envs/boson_peak_analysis
cd /home/2/kj405942/POLFED_bosons/bose_hubbard_star_L7_fast
chmod u+rwx .
chmod u+x pbs_worker.sh pbs_analyze.sh submit_pbs.sh
python -m unittest discover -s tests -v
python run_l7_star.py validate --config config_L7_corrected.json
bash submit_pbs.sh config_L7_corrected.json
```

Sprawdzenie:

```bash
qstat -u kj405942
python run_l7_star.py status --config config_L7_corrected.json
grep -R -n -E 'Traceback|Error|Killed|MemoryError|walltime' logs || true
```

Po komunikacie `Complete 72/72; missing 0`:

```bash
qsub -l walltime=01:00:00,mem=8gb -o "$PWD/logs/analyze_corrected.out" \
  -v CONFIG_PATH="$PWD/config_L7_corrected.json",PROJECT_DIR="$PWD",PYTHON_BIN="$(command -v python)" \
  pbs_analyze.sh
```

Analiza zapisuje m.in. `peaks.csv`, `fits.csv`, `direct_fits.csv`,
`channels.csv`, rozkład lokalnej entropii, skompresowany surowy CSV,
`summary.txt` oraz PNG/PDF. Maksima brzegowe i niestabilne nie są zamieniane na
zera — dostają `NaN` i jawną flagę jakości.

Opcjonalne porównanie z ED: dodaj do `paths` w konfiguracji
`"ed_peaks_csv": "sciezka/do/plik.csv"`. CSV powinien zawierać `N`, `nmax`,
`W_over_t` oraz `U_peak` (albo `U_S_star_over_t`). Program zapisze reszty i
RMSE bez dopasowywania modelu do danych ED.

## GitHub

Surowych `task_*.npz` nie należy wysyłać. Z katalogu repozytorium:

```bash
cd /home/2/kj405942/POLFED_bosons
git add bose_hubbard_star_L7_fast
git add -f bose_hubbard_star_L7_fast/results_L7_corrected/analysis
git status --short
git commit -m "Correct L7 Bose-Hubbard star-model analysis"
git push origin HEAD
```

Przed commitem sprawdź, czy na liście nie ma katalogu
`results_L7_corrected/raw`. Jest ignorowany przez `.gitignore`.
