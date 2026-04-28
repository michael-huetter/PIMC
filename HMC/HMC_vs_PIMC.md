# HMC vs PIMC erster Vergleich


## Autokorrelationsplots

### Autokorrelation von HMC bei T = 0.5
![T = 0.5 ](Bilder/HMC_autokorr_T0.5.png)

### Autokorrelation von HMC bei T = 10.0
![T = 10.0 ](Bilder/HMC_autokorr_T10.0.png)

### Autokorrelation von PIMC bei T = 0.5
![T = 0.5 ](Bilder/PIMC_autokorr_T0.5.png)

### Autokorrelation von PIMC bei T = 10.0
![T = 10.0 ](Bilder/PIMC_autokorr_T10.0.png)

#### $\tau_{int}$ beschreibt die mittlere Zeit (Anzahl der Schritte) bis die Messdaten wieder unabhänig von einander sind.

#### es fällt auf das vor allem die Autokorrelation der Potenziellen Energie sowohl beim HMC und bei PIMC mit der Temperatur stark ansteigen. Somit wird die effektive Anzahl der Messpunten kleiner und die Unsicherheit der Potenziellen Energie steiht mit der Temperatur an.

#### im vergleich hat der HMC eine geringeres $\tau_{int}$ bei der Kinetischen Energie wie der PIMC, diese ändert sich mit der Temperatur auch kaum. Dafür ist das $\tau_{int}$ der Poteziellen Energie und somit auch der Position größer wie bei PIMC.



## HMC vs PIMC Mittelwert über Rechenzeit

#### HMC T = 0.5

![T = 0.5 ](Bilder/HMC_Ekin_mean_Zeit_T0.5.png)
![T = 0.5 ](Bilder/HMC_Epot_mean_Zeit_T0.5.png)

#### PIMC T = 0.5

![T = 0.5 ](Bilder/PIMC_Ekin_mean_Zeit_T0.5.png)
![T = 0.5 ](Bilder/PIMC_Epot_mean_Zeit_T0.5.png)

## Observalenmittelwerte über Temperatur

### Potentielle und Kinetische Energie Mittelwerte mt HMC bestimmt.
![HMC](Bilder/HMC_E_T.png)

### Potentielle und Kinetische Energie Mittelwerte mt PIMC bestimmt.
![PIMC](Bilder/PIMC_E_T.png)

#### laut Viralsatz erwartet man das bei einem Harmonischen Potenzial die Kinetische Energie gleich der Potenziellen Energie entspricht dies ist bei niedrigen Temperaturen auch noch sehr genau erfüllt, bei hohen Temperaturen steigt die Unsicherheit der Potenziellen Energie stark und man sieht sowhol beim HMC und bei PIMC eine systematische Abweichung der Potenziellenenegie nach unten.

#### wärend bei HMC die Unsicherheiten der Potenziellen Energie bei höheren Temperaturen (T =8, 10) etwas größer sind wie beim PIMC, sind aber die unsicherheiten der Kinetischen Energie bei PIMC viel kleiner sodass man die Fehlerbalken nicht im Plot erkennen kann

## Beispielplots der kinetischen und Potenziellen Energie vom PIMC
### $N$ = 10000 (Montecarlo-Steps)

![T = 0.5 ](Bilder/PIMC_T0.5.png)
![T = 4.0 ](Bilder/PIMC_T4.0.png)
![T = 10.0 ](Bilder/PIMC_T10.0.png)

## Beispielplots der kinetischen und Potenziellen Energie vom HMC
### Parameter: $\Delta T$ = 0.03 ,  $L$ = 100 (Leapfrog-Steps), $N$ = 10000 (Montecarlo-Steps) ,   $\mu$ = 2

![T = 0.5 ](Bilder/HMC_T0.5.png)
![T = 4.0 ](Bilder/HMC_T4.0.png)
![T = 10.0 ](Bilder/HMC_T10.0.png)