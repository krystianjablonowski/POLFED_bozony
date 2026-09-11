# Szybki model gwiazdy Bosego–Hubbarda dla L=7

To jest niezależna wersja 2.1 programu. Nie czyta wyników poprzedniej wersji i
nie wykonuje pełnej diagonalizacji Hamiltonianu wielu ciał. Diagonalizuje tylko
lokalne macierze gwiazdy o rozmiarze najwyżej `13 x 13`, grupami w jednej
wektorowej operacji NumPy.

## Co dokładnie liczy pilot

- `L=7`, otwarty łańcuch, `t=1`;
- sektory `N=nmax=4,5,6,7`, zgodnie z czterema panelami danych ED;
- Hamiltonian diagonalny
  `U*sum_i n_i(n_i-1) + sum_i epsilon_i*n_i`;
- niezależne `epsilon_i` jednostajne w `[-W,W]`;
- `W/t=0.1,...,2.5` i `U/t=0,...,0.6` z krokiem `0.005`;
- 30 realizacji nieporządku i maksymalnie 128 konfiguracji ze środkowej połowy
  uporządkowanych energii diagonalnych;
- 100 niezależnych zadań PBS: po jednym dla każdego sektora i wartości `W`.

Pilot ma sprawdzić kształt i położenie maksimum. Dopiero po jego obejrzeniu
należy zwiększać statystykę. Graniczne maksima nie są zapisywane jako zero:
mają `U_peak=nan` oraz jawną flagę jakości.

Ważne: to jest test hipotezy, a nie dopasowanie do danych ED. Program nie ma
parametru przesuwającego maksimum do żądanej wartości. Jeżeli model gwiazdy
głębokości 1 nie odtworzy krzywej ED, wynik ma to pokazać wprost.

## Test lokalny

```bash
python -m unittest discover -s tests -v
python run_l7_star.py validate --config config_L7_pilot.json
```

## Wysłanie z Windows na Kruka

W PowerShellu lub CMD przejdź do katalogu `POLFED`, a następnie:

```text
scp -r bose_hubbard_star_L7_fast kj405942@kruk-host.fuw.edu.pl:/home/2/kj405942/POLFED_bosons/
```

## Uruchomienie na Kruku

```bash
source /home/2/kj405942/conda_envs/boson_peak_analysis/bin/activate
cd /home/2/kj405942/POLFED_bosons/bose_hubbard_star_L7_fast
chmod u+x pbs_worker.sh pbs_analyze.sh submit_pbs.sh cleanup_old_results_on_kruk.sh
python -m unittest discover -s tests -v
python run_l7_star.py validate --config config_L7_pilot.json
bash submit_pbs.sh
```

Skrypt wypisze identyfikator tablicy PBS. Wszystkie 100 elementów tablicy jest
zgłoszonych bez limitu równoczesności po stronie użytkownika; faktyczną liczbę
uruchomionych naraz wybiera scheduler Kruka.

Stan obliczeń:

```bash
qstat -u kj405942
python run_l7_star.py status --config config_L7_pilot.json
grep -R -n -E 'Traceback|Error|Killed|MemoryError|walltime' logs || true
```

Gdy `status` pokaże `Complete 100/100`, uruchom krótką analizę i wykresy:

```bash
qsub -l walltime=00:30:00,mem=2gb -o "$PWD/logs/analyze.out" \
  -v CONFIG_PATH="$PWD/config_L7_pilot.json",PROJECT_DIR="$PWD",PYTHON_BIN="$(command -v python)" \
  pbs_analyze.sh
```

Po zakończeniu powinny istnieć `results_L7_pilot/analysis/peaks.csv`,
`curves.csv`, `summary.txt` oraz figury PNG/PDF. Analiza odmawia startu, jeśli
choć jedno zadanie jest brakujące albo uszkodzone.

## Usunięcie błędnych wyników poprzedniej wersji

Ta operacja jest nieodwracalna. Skrypt pokazuje rozmiar i wymaga wpisania
`DELETE`; usuwa tylko `results_L7_L8` i stare logi, a zachowuje kod źródłowy:

```bash
bash cleanup_old_results_on_kruk.sh
```

## GitHub po weryfikacji pilota

Wyniki są domyślnie ignorowane przez Git, żeby nie wysłać dziesiątek megabajtów
surowych plików. Najpierw dodaj sam kod i małe tabele/figury analizy:

```bash
cd /home/2/kj405942/POLFED_bosons
git add bose_hubbard_star_L7_fast ':!bose_hubbard_star_L7_fast/results_L7_pilot/raw'
git add -f bose_hubbard_star_L7_fast/results_L7_pilot/analysis
git commit -m "Add fast L7 Bose-Hubbard star-model pilot"
git push origin HEAD
```

Przed `git commit` warto wykonać `git status --short` i sprawdzić, czy w staged
changes nie ma surowych plików `task_*.npz`.

## Test zbieżności po pilocie

`config_L7_convergence.json` liczy pięć kontrolnych wartości
`W/t=0.8,1.2,1.6,2.0,2.5`, 300 realizacji oraz wszystkie konfiguracje ze
środkowej połowy energii. Powstaje 60 zadań (4 sektory x 5 wartości W x 3
części po 100 realizacji):

```bash
bash submit_pbs.sh config_L7_convergence.json
python run_l7_star.py status --config config_L7_convergence.json
```

Analizę należy uruchomić dopiero po `Complete 60/60; missing 0`, z limitem
jednego wątku OpenBLAS. Odrzucone kandydaty maksimum są oznaczone krzyżykami;
nie są łączone linią ani przedstawiane jako poprawne maksima.
