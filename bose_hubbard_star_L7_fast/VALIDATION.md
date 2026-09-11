# Raport walidacji wersji 3.0.0

Polecenie `python -m unittest discover -s tests -v` przechodzi 15/15 testów.

Sprawdzone zostały:

- wymiary baz `L=7` dla `N=nmax=4,5,6,7`;
- zgodność diagonalizacji wektorowej ze skalarną implementacją referencyjną;
- dokładne przypadki jednej krawędzi, rezonansu i zerowego sprzężenia;
- zgodność bezpośredniego detuningu z formułą kanałową;
- norma każdego wektora własnego do tolerancji `1e-12`;
- dokładna zgodność realizacji nieporządku z konwencją seeda kodu ED;
- wybór 20 konfiguracji najbliższych znormalizowanej energii `0.5`;
- zachowanie wcześniejszej definicji `M` i jej niezależne przeliczenie;
- równość macierzy gwiazdy i odpowiedniego lokalnego bloku niezależnie
  zbudowanego pełnego Hamiltonianu `L=N=4`;
- uruchomienie pełnej ED i modelu gwiazdy na tej samej małej realizacji;
- niezmienniczość wszystkich głównych obserwabli na permutację kolejności bazy;
- odzyskanie znanego maksimum paraboli oraz test stabilności okna 3/5/7.

Test end-to-end dla trzech wartości `W`, trzech realizacji i sektora `N=4`
zakończył się poprawnie: powstały surowe NPZ, długi skompresowany CSV, tabele
krzywych/maksimów/kanałów/rozkładu oraz wykresy PNG i PDF.

Poprzednich wyników wersji 2.x nie wolno łączyć z wersją 3.0. Plik każdego
zadania zawiera wersję i SHA-256 części obliczeniowej konfiguracji; analiza
odrzuca brakujące, stare albo niezgodne pliki.

Wersja 3.0 nie gwarantuje z góry, że fizyczny model głębokości 1 odtworzy ED.
Gwarantuje natomiast, że brak maksimum nie będzie skutkiem podmiany definicji
`M`, innego seeda, błędnego okna centralnego, globalnej zamiast lokalnej
normalizacji lub cichego wpisania zera dla odrzuconego maksimum.
