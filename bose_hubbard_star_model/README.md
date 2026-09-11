# Model gwiazdy dla nieuporządkowanego modelu Bosego–Hubbarda

Projekt wyznacza maksima `M`, `S2` i `Sstar` bez budowania i bez
diagonalizacji pełnego Hamiltonianu wielu ciał. Diagonalizowane są wyłącznie
lokalne macierze gwiazdy o wymiarze `1 + z`.

## Konwencje zgodne z istniejącym kodem POLFED

```text
H = -t sum_<i,j> (b_i^dagger b_j + h.c.)
    + U sum_i n_i(n_i-1) + sum_i epsilon_i n_i
epsilon_i ~ Uniform[-W,W], open boundary
```

Każdy `sample_id` ma ten sam bezwymiarowy wektor nieporządku dla wszystkich
`U` i `W`; dla danego `W` używane jest `epsilon=W*x`. Zapewnia to parowane
porównania. Centralne konfiguracje są wybierane z połówki środka
uporządkowanego widma energii diagonalnych, co odpowiada dotychczasowemu
`middle_fraction=0.5`. Liczbę konfiguracji ogranicza parametr
`n_configurations`.

## Test lokalny

```bash
python3 -m unittest discover -s tests -v
python3 star_model.py validate --config config_smoke.json
python3 star_model.py init --config config_smoke.json --force
python3 star_model.py worker --config config_smoke.json --task-id 0
python3 star_model.py worker --config config_smoke.json --task-id 1
python3 star_model.py worker --config config_smoke.json --task-id 2
python3 star_model.py worker --config config_smoke.json --task-id 3
python3 star_model.py worker --config config_smoke.json --task-id 4
python3 star_model.py worker --config config_smoke.json --task-id 5
python3 star_model.py aggregate --config config_smoke.json
```

Komenda `status` pokazuje brakujące identyfikatory:

```bash
python3 star_model.py status --config config_smoke.json
```

## Wysłanie na Kruka

Z Windows PowerShell:

```powershell
scp -r "C:\Users\avoga\OneDrive\Dokumenty\POLFED\bose_hubbard_star_model" `
  kj405942@kruk-host:/home/2/kj405942/POLFED_bosons/
```

Na Kruku:

```bash
cd /home/2/kj405942/POLFED_bosons/bose_hubbard_star_model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python star_model.py init --config config_L7_L8.json
bash submit_pbs.sh config_L7_L8.json all
```

Przed produkcją warto zmierzyć czas jednego zadania:

```bash
python star_model.py worker --config config_L7_L8.json --task-id 0
```

Jeżeli czas przekracza limit kolejki, zmniejsz `samples_per_task` (produkcja
ma domyślnie 100). Jeżeli jest bardzo krótki, można ten parametr zwiększyć. Plik wynikowy istniejącego
zadania nie jest nadpisywany, więc ponowne wysłanie jest bezpieczne.

Po ukończeniu wszystkich zadań:

```bash
python star_model.py status --config config_L7_L8.json
qsub -v CONFIG_PATH=config_L7_L8.json pbs_aggregate.sh
```

## Wyniki

W `results_L7_L8/analysis/` powstaną:

- `raw_observables.csv.gz` — skompresowany CSV z wierszem dla każdego `(L,N,nmax,W,U,sample)`;
- `channels.csv.gz` — skompresowany CSV z rozkładem `M_k`, `S2_k` i udziałem krawędzi `P_k`;
- `Sstar_distribution.csv.gz` — histogram lokalnych wartości entropii gwiazdy;
- `peaks.csv` — lokalne dopasowania paraboliczne i bootstrapowe przedziały;
- `convergence_peaks.csv` — maksima dla prefiksów 100 i 300 realizacji;
- `fits.csv` — dopasowania `U*=b+cW` i `eta=c_X/c_M`;
- `summary.txt`;
- `figures/*.png` i `figures/*.pdf`.

Surowe, wznawialne pliki każdego zadania są w `results_L7_L8/raw/`.

## Koszt i ustawienia produkcyjne

`config_L7_L8.json` używa kroku `dU=0.005`, 300 realizacji i 300 centralnych
konfiguracji. To jest kosztowny, lecz nadal lokalny rachunek. Najpierw należy
uruchomić `config_smoke.json`, potem jedno zadanie produkcyjne. Stabilność
można sprawdzić powtarzając agregację dla prefiksów 100 i 300 realizacji;
wariant 1000 realizacji warto uruchamiać dopiero, jeśli maksimum nie jest
stabilne w granicach żądanej dokładności.

Model nie wczytuje ani nie liczy ED. Plik `peaks.csv` jest przeznaczony do
bezpośredniego połączenia z istniejącą tabelą maksimów entropii ED.

## Problem `Permission denied` po skopiowaniu na Kruka

Jeżeli aktywne jest już środowisko Conda zawierające NumPy i Matplotlib, nie
trzeba tworzyć `.venv`. Brak możliwości utworzenia zarówno `.venv`, jak i
`results_L7_L8` oznacza brak prawa zapisu do katalogu projektu. Sprawdź i
napraw prawa właściciela:

```bash
cd /home/2/kj405942/POLFED_bosons/bose_hubbard_star_model
ls -ld . ..
chmod u+rwx .
mkdir -p results_L7_L8/raw results_L7_L8/analysis/figures logs
```

Jeżeli `chmod` zwróci `Operation not permitted`, katalog nie należy do
bieżącego użytkownika. Wtedy skopiuj kod do nowego katalogu należącego do
użytkownika i pracuj w kopii:

```bash
cd /home/2/kj405942/POLFED_bosons
cp -r bose_hubbard_star_model bose_hubbard_star_model_work
chmod -R u+rwX bose_hubbard_star_model_work
cd bose_hubbard_star_model_work
```
