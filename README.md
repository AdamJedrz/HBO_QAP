# QAP HBO

Projekt implementuje algorytm **Heap-Based Optimizer (HBO)** dla problemu **Quadratic Assignment Problem (QAP)**.

Program wczytuje instancje QAP, uruchamia algorytm optymalizacji i porównuje uzyskane wyniki z najlepszymi znanymi rozwiązaniami, jeśli są dostępne.

## Struktura projektu

```text
.
├── HBO.py
├── qapdata/
│   └── pliki .dat z instancjami QAP
└── qapsoln/
    └── pliki z najlepszymi znanymi rozwiązaniami
```

## Wymagania

Projekt wymaga Pythona 3 oraz biblioteki numpy.

Instalacja zależności:

```text
pip install numpy
```

## Uruchomienie

Aby uruchomić program:

```text
python HBO.py
```

Domyślnie program wczytuje wybrane instancje z folderu qapdata i wykonuje dla nich test algorytmu HBO.

## Dane wejściowe

Instancje problemu powinny znajdować się w folderze:

```text
qapdata/
```

Rozwiązania referencyjne znajdują się w folderze:

```text
qapsoln/
```

Program obsługuje pliki w formacie QAPLIB.
