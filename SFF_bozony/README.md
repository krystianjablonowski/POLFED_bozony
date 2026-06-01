# SFF bozonow i czas Thoulessa

Ten folder zawiera programy do analizy spectral form factor (SFF) i czasu
Thoulessa dla nieuporzadkowanego lancucha Bose-Hubbarda. Analiza korzysta z
widm zapisanych przez:

```text
../run_bose_hubbard_polfed_N_fixed_nmax.jl
```

Skrypt rozpoznaje zarowno pliki `energies_polfed_*.txt`, jak i
`energies_full_*.txt`. Kazda kolumna jest osobna realizacja nieporzadku, a
wiersze zawieraja poziomy energii. Parametry `L`, `N`, `nmax`, `U`, `W` i
`boundary` sa odczytywane bezposrednio z nazwy pliku.

## Uruchomienie

Z glownego folderu `POLFED_bosons`:

```bash
python SFF_bozony/compute_sff_thouless_bosons.py \
  --input-dir pbs_bose_polfed_grouped/data
```

Wyniki sa domyslnie zapisywane w `SFF_bozony/results`:

- `thouless_summary.csv`: zbiorcza tabela `tau_Th`, sredniego odstepu
  poziomow, `t_H`, `t_Th` i `g`;
- `curves/*.txt`: dane SFF, GOE i `DeltaK` dla kazdego punktu;
- `plots/*.png` i `plots/*.pdf`: wykresy SFF i `DeltaK`;
- `plots/thouless_vs_W_L*_N*_nmax*_boundary*.*`: zbiorcze wykresy czasow
  wzgledem `W`;
- `plots/g_vs_W_U*.*`: wykresy wskaznika ergodycznosci `g(W)`.

Do obliczen wystarcza `numpy`. Wykresy wymagaja `matplotlib`. Jezeli
`matplotlib` nie jest dostepny, skrypt nadal zachowa tabele i krzywe tekstowe.

## Definicje

Po lokalnym unfoldingu srednia odleglosc miedzy poziomami wynosi jeden.
Gaussowskie okno energetyczne ogranicza wplyw brzegow probki POLFED:

```text
K(tau) = |sum_n w_n exp(-2 pi i tau epsilon_n)|^2 / sum_n w_n^2
tau = t / t_H
t_H = 2 pi / <delta E>
DeltaK = |log10(K / K_GOE)|
```

`tau_Th` jest pierwszym stabilnym wejsciem krzywej SFF w otoczenie GOE po
minimum correlation hole. Domyslnie wymagane sa cztery kolejne punkty z
`DeltaK < 0.08`. Program zapisuje tez:

```text
t_Th = tau_Th * t_H
g = log10(t_H / t_Th) = -log10(tau_Th)
```

## Wybieranie danych

Skrypt domyslnie pomija pliki `_partial`. Jezeli dla tego samego punktu
`(L, N, nmax, boundary, U, W)` istnieje kilka plikow, automatycznie wybiera
plik z najwiekszym `nreal` zapisanym w nazwie.

Liste wybranych plikow mozna sprawdzic przed analiza:

```bash
python SFF_bozony/compute_sff_thouless_bosons.py \
  --input-dir pbs_bose_midpoint_L9/data \
  --L-list 9 \
  --N 9 \
  --nmax 2 \
  --boundary periodic \
  --U-list 1,5,10 \
  --W-list 2,6,10 \
  --list-files-only
```

## Przyklady

Analiza siatki `L=9`, `N=9`, `nmax=2`, bez generowania setek
indywidualnych wykresow:

```bash
python SFF_bozony/compute_sff_thouless_bosons.py \
  --input-dir pbs_bose_midpoint_L9/data \
  --output-dir SFF_bozony/results_nmax2_L9 \
  --L-list 9 \
  --N 9 \
  --nmax 2 \
  --boundary periodic \
  --middle-count 150 \
  --min-levels 50 \
  --skip-individual-plots
```

Analogiczna analiza danych z wiekszym dopuszczalnym obsadzeniem wezla:

```bash
python SFF_bozony/compute_sff_thouless_bosons.py \
  --input-dir study_nmax3/pbs_bose_midpoint_L8/data \
  --output-dir SFF_bozony/results_nmax3_L8 \
  --L-list 8 \
  --N 8 \
  --nmax 3 \
  --boundary periodic \
  --middle-count 150 \
  --min-levels 50 \
  --skip-individual-plots
```

Dla `nmax=4` wystarczy zmienic katalogi i parametr:

```bash
python SFF_bozony/compute_sff_thouless_bosons.py \
  --input-dir study_nmax4/pbs_bose_midpoint_L8/data \
  --output-dir SFF_bozony/results_nmax4_L8 \
  --L-list 8 \
  --N 8 \
  --nmax 4 \
  --boundary periodic \
  --middle-count 150 \
  --min-levels 50 \
  --skip-individual-plots
```

Zakres mozna ograniczyc opcjami `--min-U`, `--max-U`, `--min-W`,
`--max-W`, a konkretne wartosci podac przez `--U-list` i `--W-list`.

## Skalowanie skonczonego rozmiaru: odpowiednik Fig. 7

Skrypt `plot_thouless_scaling_figure7.py` korzysta z tabel
`thouless_summary.csv`. Nie przelicza SFF i nie uruchamia diagonalizacji.
Dla par rozmiarow `L1 < L2` przy tych samych parametrach liczy:

```text
z(W)     = log(t_Th(L2) / t_Th(L1)) / log(L2 / L1)
xi_Th(W) = (L2 - L1) / log(t_Th(L2) / t_Th(L1))
```

Program laczy wylacznie dane o tym samym `nmax`, `boundary` i wypelnieniu
`N/L`. Nie pomiesza wiec przypadkow `nmax=2`, `nmax=3` i `nmax=4`.

Przyklad dla `nmax=2` i rozmiarow `L=7,8,9`:

```bash
python SFF_bozony/plot_thouless_scaling_figure7.py \
  --summary SFF_bozony/results_nmax2_L7/thouless_summary.csv \
  --summary SFF_bozony/results_nmax2_L8/thouless_summary.csv \
  --summary SFF_bozony/results_nmax2_L9/thouless_summary.csv \
  --output-dir SFF_bozony/figure7_scaling_nmax2_L7_L8_L9 \
  --L-list 7,8,9 \
  --nmax-list 2 \
  --boundary periodic
```

Konkretny wykres, na przyklad tylko dla `U=2`, mozna wybrac przez
`--U-list 2`.
