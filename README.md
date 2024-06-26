# Introduzione
Questa è la relazione sul progetto di Machine Learning di Alex Bonini (559298), per il corso tenuto nell'anno accademico 2023/2024.
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
In [4-modello-passi-CNN](scripts/4-modello-passi-CNN.ipynb) ho usato un semplice modello convoluzionale per addestrarlo a prevedere l'andamento dei passi. Questo è servito per testare la capacità del modello di apprendere i pattern interni all'andamento dei passi. In più questo può essere utilizzato come punto di partenza per un modello più complesso che riesce anche a lavorare con la qualità del sonno. I risultati sono già promettenti, il modello riesce sia a seguire l'andamento che a prevederlo, come si può vedere dai grafici all'interno del notebook.

In [4-modello-passi-LSTM](scripts/4-modello-passi-LSTM.ipynb) ho usato un modello ricorrente con gli stessi dati del tentativo precedente. Anche questo riesce a seguire molto bene l'andamento e si aggancia bene alle variazioni periodiche. Forse però le sue previsioni sono leggermente meno realistiche rispetto al modello precedente.

In [5-modello-sonno-mlpregressor](scripts/5-modello-sonno-mlpregressor.ipynb) ho preso il modello convoluzionale già addestrato sui passi, ho estratto le attivazioni dell'ultimo layer convoluzionale e poi ho usato queste attivazioni per addestrare un percettrone multistrato a prevedere la qualità del sonno. I risultati mostrano che questo modello riesce a capire il tipo di andamenti e soprattutto la varianza della qualità del sonno. Purtroppo però il valore assoluto delle predizioni non è sempre accurato.

In [5-modello-sonno-ridge](scripts/5-modello-sonno-ridge.ipynb) ho fatto la stessa cosa del modello precedente, ma ho utilizzato una Ridge. Il motivo è che lo scopo di usare un modello già addestrato è proprio quello di poterci mettere sopra dei modelli più semplici e veloci. La Ridge inoltre ha una normalizzazione che dovrebbe aiutare il modello ad essere meno influenzato dal rumore di cui abbiamo parlato prima. Purtroppo però come si può vedere dai grafici, nonostante aver provato con diversi valori del coefficiente della normalizzazione, la Ridge tende comunque a fare previsioni sempre vicinissime al valore medio e che quindi risultano "piatte". Questo probabilmente è sempre dovuto al rumore che c'è nei dati che non permette ad un modello così semplice di trovare pattern, quindi si limita a predire la media per minimizzare l'errore.

In [5-modello-sonno](scripts/5-modello-sonno.ipynb) ho preso il modello convoluzionale già addestrato, poi ho sostituito l'ultimo layer (quello che genera il valore finale, cioè il numero di passi) con uno uguale, che poi però è stato riaddestrato per trovare la qualità del sonno. In questo modo ho beneficiato di un modello parzialmente addestrato nel riconoscere i pattern dei passi, che è già gran parte del lavoro necessario per predire la qualità del sonno. Dai risultati si vede che riesce a predire meglio della Ridge e spesso individua correttamente i trend positivi e negativi. Capisce le oscillazioni e spesso è abbastanza accurato anche dal punto di vista del valore assoluto.

In [6-passi-sonno-CNN-diff](scripts/6-passi-sonno-CNN-diff.ipynb) ho preso un modello convoluzionale nuovo e l'ho addestrato sui delta tra medie consecutive di cui si è parlato sopra. La speranza era che il modello riuscisse a capire, in base all'andamento dei passi nei giorni precedenti, se la qualità del sonno doveva aumentare o diminuire. Purtroppo però come si vede dal grafico i delta (nonostante fossero fatti sui valori medi, quindi con meno rumore) sono troppo rumorosi e farne i delta non ha fatto che accentuare le piccole variazioni che sono spesso causate dal rumore.

In [6-passi-sonno-CNN](scripts/6-passi-sonno-CNN.ipynb) ho preso un modello convoluzionale fresco e l'ho addestrato subito dandogli i dati sui passi in input e quelli sul sonno in output. Questo è, come si poteva intuire, quello che ha ottenuto i risultati migliori.

In [6-passi-sonno-LSTM-diff](scripts/6-passi-sonno-LSTM-diff.ipynb) ho provato la stessa idea dei delta ma con un modello ricorrente. Purtroppo però il problema è proprio nella rumorosità dei dati e non si sono ottenuti risultati entusiasmenti.

In [6-passi-sonno-LSTM](scripts/6-passi-sonno-LSTM.ipynb) ho provato con un modello ricorrente fresco e l'ho addestrato con i passi e la qualità del sonno. Anche qui si sono ottenuti risultati molto buoni, con un modello che fa predizioni molto sensate e ragionevoli. L'esito è paragonabile e forse leggermente migliore di quello ottenuto con il modello convoluzionale. Dopotutto le reti ricorrenti sono particolarmente adatte a questo tipo di predizioni...

# Visualizzazioni
La visualizzazione dei dati è un'altra parte importante e molto utile del mini-framework. Visto che è stata fatta per il progetto, ho ritenuto giusto parlarne un po' qui. Se non altro per spiegare i significati dei vari grafici. 

Anche in questo caso lo strumento permette di fare grafici con grande facilità. Un elemento chiave è che permette molto facilmente di costruire il grafico su dati specifici (basta istanziare una nuova classe dei dati), in questo modo si può evitare di usare dati che il modello ha già visto o fare una qualche ricerca di dati più rappresentativi del comportamento del modello.
Tra le visualizzazioni implementate abbiamo:
- DateStepForecastVisualization: Mostra il grafico con la data sulla x e i passi sulla y. Mostra la curva del forecast, cioè una previsione del futuro fatta concatenando le previsioni precedenti e riusandole in input.
- DateStepPredictionVisualization: Mostra il grafico di prima, ma mostra la predizione invece del forecast. La predizione è fatta sempre basandosi su una finestra di dati reali, quindi inventa solo il punto corrente.
- DateStepPredictionForecastVisualization: Questo mostra sia predizione che forecast.
- DateSleepPredictionVisualization: Questo grafico è simile al precedente, ma mostra la predizione sulla qualità del sonno invece che sui passi. Bisogna tenere a mente però che le predizioni sono tutte fatte basandosi solo sui passi (non sui dati del sonno).
- DateSleepStepPredictionVisualization: Mostra sotto i dati dei passi, cioè quelli usati per la predizione. Sopra invece mostra i dati dei passi e la relativa predizione.
- DateMeanSleepStepPredictionVisualization: Come quello sopra, ma mostra anche la curva delle medie ottenute con sliding window.
- SleepStepPredictionVisualization: Simile a DateSleepPredictionVisualization, ma sulla x non c'è la data, c'è solo l'indice della previsione.

# Conclusioni
I risultati e le difficoltà sono stati già discussi sopra, quindi questa parta è dedicata soprattutto alle considerazioni personali.

Questo progetto è stato la mia prima avventura nel mondo del machine learning. Ho avuto modo di spaziare su una grandissima varietà di approcci e metodi. Nonostante la difficoltà dell'obiettivo si è riusciti anche ad ottenere risultati tutto sommato soddisfacenti per un progetto didattico. Questa breve relazione mostra solo i risultati finali e nasconde un lungo processo di apprendimento (training, se vogliamo...) che però porterò con me anche nei prossimi progetti.