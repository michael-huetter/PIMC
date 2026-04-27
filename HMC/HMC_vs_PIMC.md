# HMC vs PIMC erster Vergleich

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

## Autokorrelationsplots und Mittelwert über Rechenzeit

### HMC vs PIMC Rechenzeit

#### HMC T = 0.5

![T = 0.5 ](Bilder/HMC_Ekin_mean_Zeit_T0.5.png)
![T = 0.5 ](Bilder/HMC_Epot_mean_Zeit_T0.5.png)

#### PIMC T = 0.5

![T = 0.5 ](Bilder/PIMC_Ekin_mean_Zeit_T0.5.png)
![T = 0.5 ](Bilder/PIMC_Epot_mean_Zeit_T0.5.png)

#### es fällt auf das ber PIMC für gleich viele Montecarlo schritte viel weniger Rechenzeit braucht. Wenn man jedoch auf die Unsicherheiten der Mittelwerte schaut sieht man das der HMC bereits nach ca. 3000 Schritten ähnliche Unsicherheiten hat wie der PIMC und sich somit die Rechen Zeiten mit 5.5s beim HMC und 4s beim PIMC für die selben Unsicherheiten nicht mehr so stark unterscheiden. 
## Observalenmittelwerte über Temperatur
