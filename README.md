# Introduzione
## Obiettivo
Lo scopo del progetto è quello di utilizzare la serie temporale dei passi di un utente per cercare di fare previsioni sul suo benessere, indicato dalla qualità del sonno.
Ho quindi ottenuto una grande quantità di dati sui passi e su metriche relative al sonno. 
Ho poi elaborato questi dati per ricavarne metriche utili e renderli più facilmente utilizzabili da un modello di machine learning.
Per rendere più facile sperimentare con diverse combinazioni di dati e modelli, ho programmato una sorta di [mini-framework](scripts/shared_utilities.py) che mi permettesse di lavorare in maniera più "dichiarativa".
## Sfida
La sfida più grande del progetto è stata affrontare il grandissimo rumore presente all'interno dei dati che riguardano le metriche del sonno. 
Questo è dovuto in parte alla qualità dei dati, che essendo ricavati da sensori economici hanno spesso buchi e valori altamente "sballati".

Un altro fattore di assoluta rilevanza è la correlazione tra passi e sonno. 
Infatti è vero che uno stile di vita attivo porta ad un maggior benessere, che poi si traduce in una migliore qualità del sonno.
Il problema è che la qualità del sonno è a sua volta influenzata da innumerevoli altri elementi sia psicologici che fisici che non sono rappresentati dall'andamento dei passi.
## Risultati
Nonostante la gravità della sfida, abbiamo già appurato che esiste una qualche dipendenza tra la qualità del sonno e l'andamento dei passi.
Applicando certi accorgimenti, sono riuscito ad ottenere un modello che riuscisse a palesare questo rapporto.

Nel farlo ho anche avuto modo di familiarizzare con gli strumenti principali del machine learning:
- Strumenti di elaborazione dati
  - Pandas
  - Numpy
  - Matplotlib
  - Standardizzazione
  - Normalizzazione
  - Sliding windows
- Keras
  - CNN
  - RNN (LSTM)
  - Reti multistrato in generale
  - Analisi e utilizzo delle attivazioni dei layer di una rete neurale
- Scikit-Learn
  - Ridge
  - MLP
# I dati
## Processamento
Il dataset originale è sparso e in [0-converti-dati](scripts/0-converti-dati.ipynb) è possibile vedere cosa ho fatto per ottenerne una versione densa con solo i dati che interessano a me, cioè quelli sui passi e le metriche del sonno.

Ho poi analizzato le metriche a disposizione in [1-analisi-features](scripts/1-analisi-features.ipynb), dove ho visto che la REM duration purtroppo manca quasi sempre e non conviene usarla.

In [2-punteggio-sonno](scripts/2-punteggio-sonno.ipynb) è possibile vedere come ho utilizzato le metriche del sonno rimaste per ottenere una stima molto ragionevole della qualità del sonno dell'utente.
Ho calcolato questo punteggio considerando il prodotto tra tre diverse penalità:
- Scostamento dalla percentuale ottima di sonno profondo
- Scostamento dalla percentuale ottima di sonno leggero
- Scostamento dalla durata media del sonno dell'utente
I punteggi del sonno usciti da questa valutazione erano molto buoni, ma purtroppo l'andamento della qualità del sonno dell'utente risentiva del rumore del dataset.

Per migliorare quindi i dati e renderli utilizzabili in maniera efficiente da un modello di machine learning, ho fatto alcuni passaggi in [3-normalizza-dati](scripts/3-normalizza-dati.ipynb):
- Normalizzazione della data
- Standardizzazione dei passi e della qualità del sonno (sia sulla base del singolo utente che di tutto il dataset)
- Smoothing dei dati per ridurre il rumore, fatto attraverso una sliding window che calcola la media
- Calcolo dei delta tra due medie consecutive

## Il mini-framework

Il mini-framework ha aiutato molto con i dati, perché mi ha permesso di rimuovere frammenti di codice duplicati sparsi nei vari notebook.
Adesso c'è quindi un'interfaccia dichiarativa che si preoccupa di preparare i dati che mi servono: training set, test set e output associati.
Per me è sufficiente istanziare la classe che rappresenta i dati.
A questa classe posso fornire questi parametri:
- window: La grandezza della finestra temporale
- test_ratio: La frazioni di dati da usare per il testing
- dataset: Il nome del file da cui caricare i dati
- dataset_chunk: Quante righe caricare dal dataset
- sleep_score: Quale metrica usare per il punteggio del sonno
- step_type: Quale metrica usare per i passi
- chunk_start: Da quale riga iniziare a caricare i dati

Tra le classi che posso instanziare, ho:
- DateStepData: Dove in input ho una serie di (data, passi fatti in quella data) e in output i passi fatti nel giorno successivo alla sequenza.
- StepData: Stessa cosa, ma non senza data
- DateStepSleepData: Stessa cosa, ma in output prevedo la qualità del sonno
- StepSleepData: Stessa cosa, ma senza data

Con queste opzioni posso valutare tantissime combinazioni di dati, con numeri variabili di features, finestre temporali, etc...
Non c'è nemmeno bisogno di fare modifiche al modello per farli funzionare.

# I modelli

## Nel mini-framework
Il mini-framework mi permette di usare una grande scelta di modelli:
- ComposedKerasModel: Permette di prendere un modello Keras già esistente, sostituire il layer di output con un altro e poi continuare il training con il nuovo output.
- KerasCNNModel: Un semplice modello convoluzionale multistrato
- KerasLSTMModel: Un semplice modello ricorrente
- GenericSklearnModel: Serve a incapsulare un qualsiasi modello di Sklearn dato in input
- RidgeModel: Incapsula un modello Ridge di Sklearn
- ComposedKerasSklearnModel: Prende un modello Keras già esistente e usa le attivazioni dell'ultimo layer per addestrare un qualsiasi modello Sklearn dato in input

Vari modelli supportano opzioni leggermente diverse, conviene andarne a vedere gli utilizzi nei singoli notebook per vedere qualche esempio.

Il framework permette di collegare la classe del modello con quella dei dati e di addestrarle, valutarle ed utilizzarle tutte con la stessa semplice interfaccia.

## Combinazioni usate
In [4-modello-passi-CNN](scripts/4-modello-passi-CNN.ipynb)...

In [4-modello-passi-LSTM](scripts/4-modello-passi-LSTM.ipynb)...

In [5-modello-sonno-mlpregressor](scripts/5-modello-sonno-mlpregressor.ipynb)...

In [5-modello-sonno-ridge](scripts/5-modello-sonno-ridge.ipynb)...

In [5-modello-sonno](scripts/5-modello-sonno.ipynb)...

In [6-passi-sonno-CNN-diff](scripts/6-passi-sonno-CNN-diff.ipynb)...

In [6-passi-sonno-CNN](scripts/6-passi-sonno-CNN.ipynb)...

In [6-passi-sonno-LSTM-diff](scripts/6-passi-sonno-LSTM-diff.ipynb)...

In [6-passi-sonno-LSTM](scripts/6-passi-sonno-LSTM.ipynb)...

# Visualizzazioni


# Conclusioni
