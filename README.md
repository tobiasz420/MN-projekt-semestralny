# MN projekt semestralny

## O projekcie
Zebraliśmy zdjęcia oraz informacje o obiektach na nich widocznych, czyli współrzędne przedstawiające położenie obiektów. Podczas oznaczania skupialiśmy się na dwóch typach obiektów: ludziach i samochodach. Ostatecznie modele zostały zaprojektowane tak, by rozpoznawały wyłącznie ludzi. Zebrane dane posłużyły do trenowania modeli, które miały za zadanie wykrywać wybrany obiekt na zdjęciach.

## Struktura folderu dataset
dataset/   

│   

├── images/   

│   ├── train/   

│   │   ├── image1.png   

│   │   ├── image2.png   

│   │   └── ...   

│   ├── val/   

│       ├── image341.png   

│       ├── image342.png   

│       └── ...   

│   

├── labels/   

    ├── train/   

    │   ├── image1.txt   

    │   ├── image2.txt   

    │   └── ...   

    ├── val/   

        ├── image341.txt   

        ├── image342.txt   

        └── ...   
