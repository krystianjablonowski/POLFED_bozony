# Raport walidacji wersji 2.1.0

## Testy numeryczne

Polecenie `python -m unittest discover -s tests -v` przechodzi 8/8 testów:

- dokładne wymiary baz dla `N=nmax=4,5,6,7`;
- zgodność wektorowego solvera z niezależną diagonalizacją gwiazda po gwieździe;
- `Sstar=S2=ln(2)` dla pojedynczego rezonansu oraz zero entropii dla `V=0`;
- błąd normalizacji i tożsamości detuningu poniżej `1e-12`;
- niezmienniczy wybór konfiguracji centralnych po zmianie kolejności bazy;
- odzyskanie zadanego analitycznego maksimum paraboli.

## Pomiar czasu

Na lokalnym CPU jedno zadanie pilota (`30` realizacji, `121` wartości U,
`128` konfiguracji centralnych) trwało około 6 s dla sektora `N=4` i 11--13 s
dla najcięższego sektora `N=7`. Na klastrze każde `W` i każdy sektor jest
osobnym elementem tablicy PBS.

## Kontrola naukowa dwóch skrajnych wartości W

To są wyłącznie wartości diagnostyczne z 30 realizacji, a nie wynik
produkcyjny.

- Dla `W/t=0.1` estymatory `M_sum`, `S2_sum` i `Sstar` mają maksimum przy
  brzegu `U=0` zarówno dla `N=4`, jak i `N=7`. Program poprawnie zwraca
  `U_peak=NaN, quality_flag=boundary`.
- Dla `W/t=2.5` wewnętrzne maksima `Sstar` wyniosły w przybliżeniu:
  `0.269, 0.163, 0.223, 0.290` odpowiednio dla `N=4,5,6,7`.

Pierwsza obserwacja jest istotna: pokazane dane ED mają dodatnie maksimum także
dla małego `W`, więc podstawowy model gwiazdy głębokości 1 może nie odtwarzać
tego mechanizmu. Pełny pilot na wszystkich `W` ma rozstrzygnąć, czy i gdzie
pojawia się poprawny trend. Nie jest stosowane żadne przesunięcie ani skala
dopasowana do ED.
