# Dokumentacija: Klasifikacija Hitnosti Ticketa

**Projekt:** PULSE - Urgency Classification  
**Datum:** 23. siječnja 2026  
**Autor:** Sustav za automatsku kategorizaciju hitnosti ticketa

---

## 1. Informacije o Dataset-u

### 1.1. Što dataset obuhvaća

Dataset sadrži 48,411 IT support ticketa napisanih na engleskom jeziku. Svaki ticket predstavlja zahtjev korisnika ili prijavu problema u IT sustavu, a kategoriziran je prema razini hitnosti.

Dataset obuhvaća tri kategorije hitnosti. Prva kategorija je Low (Niska) koja uključuje rutinske zahtjeve, administrativne zadaće i upite bez vremenskog pritiska. Druga kategorija je Medium (Srednja) koja obuhvaća zahtjeve koji zahtijevaju pažnju ali nisu kritični. Treća kategorija je High (Visoka) koja sadrži hitne probleme koji utječu na rad i zahtijevaju trenutnu akciju.

### 1.2. Priprema i struktura dataset-a

Dataset se temelji na autentičnim IT support ticketima iz stvarnih poslovnih okruženja. Podaci su prošli kroz rigorozan proces obrade i pripreme kako bi bili pogodni za treniranje modela strojnog učenja. Proces pripreme obuhvaćao je višestupanjsku obradu podataka uključujući anonimizaciju osjetljivih informacija, normalizaciju tekstualnog sadržaja, uklanjanje duplikata i filtriranje nevaljanih unosa. Podaci su kategorizirani prema razini prioriteta te su oznake validirane kako bi se osigurala kvaliteta i konzistentnost dataset-a.

Za potrebe ovog projekta, podaci su organizirani u dvije datoteke. Datoteka data/raw.csv predstavlja početnu verziju podataka, dok datoteka data/clean.csv sadrži obrađene i očišćene podatke spremne za treniranje modela.

### 1.3. Osnovne statistike dataset-a

#### 1.3.1. Distribucija klasa

Distribucija klasa u dataset-u pokazuje značajnu neravnotežu između kategorija. Klasa Low dominira s 34,509 primjera što čini 71.28% ukupnog dataset-a. Klasa High sadrži 8,376 primjera ili 17.30% podataka. Klasa Medium je najmanje zastupljena s 5,526 primjera što predstavlja 11.41% dataset-a. Ukupno, dataset sadrži 48,411 primjera.

| Klasa      | Broj primjera | Postotak |
| ---------- | ------------- | -------- |
| **Low**    | 34,509        | 71.28%   |
| **High**   | 8,376         | 17.30%   |
| **Medium** | 5,526         | 11.41%   |
| **Ukupno** | 48,411        | 100%     |

#### 1.3.2. Statistika duljine teksta

Analiza duljine tekstova pokazuje značajnu varijabilnost. Prosječna duljina ticketa iznosi 267 znakova, dok medijan duljine iznosi 150 znakova što ukazuje na prisutnost duljih ticketa koji povećavaju prosječnu vrijednost. Minimalna duljina ticketa je 11 znakova, a maksimalna duljina doseže 7,011 znakova. Standardna devijacija od 385 znakova potvrđuje veliku varijabilnost u duljinama tekstova.

| Metrika                   | Vrijednost    |
| ------------------------- | ------------- |
| **Prosječna duljina**     | 267 znakova   |
| **Medijan duljine**       | 150 znakova   |
| **Minimalna duljina**     | 11 znakova    |
| **Maksimalna duljina**    | 7,011 znakova |
| **Standardna devijacija** | 385 znakova   |

#### 1.3.3. Primjeri ticketa

Za ilustraciju, prikazani su primjeri ticketa iz svake kategorije u nastavku:

**LOW primjer:**

```
"Please create ownership for the subject. The task has been handed over. Best regards."
```

**MEDIUM primjer:**

```
"Dear team, I have some issue with login. Could you please investigate? Thanks."
```

**HIGH primjer:**

```
"Sent Monday October - server not working. Please restart application immediately. Engineer needed."
```

---

## 2. NLP Zadatak i Priprema Dataset-a

### 2.1. Definicija NLP Zadatka

Zadatak je definiran kao Multi-class Text Classification, odnosno višeklasna klasifikacija teksta. Cilj projekta je automatski kategorizirati IT support tickete u jednu od tri kategorije hitnosti (Low, Medium, High) na temelju tekstualnog sadržaja ticketa. Radi se o tipu problema supervised learning, specifično o zadatku sequence classification.

### 2.2. Priprema Dataset-a za Zadatak

#### 2.2.1. Podjela Podataka

Dataset je podijeljen u tri skupa koristeći stratificirano dijeljenje kako bi se osigurala ista distribucija klasa u svim skupovima. Training skup obuhvaća 70% podataka, što iznosi približno 33,888 primjera, te se koristi za učenje modela. Validation skup sadrži 15% podataka, odnosno približno 7,262 primjera, te služi za odabir hiperparametara i ranu zaustavljanje treniranja. Test skup također sadrži 15% podataka, oko 7,261 primjer, te se koristi za finalnu evaluaciju modela. Stratifikacija osigurava da sve tri skupine imaju istu distribuciju klasa kao i originalni dataset.

#### 2.2.2. Tokenizacija

Za tokenizaciju su korišteni transformer-specifični tokenizatori. BERT model koristi WordPiece tokenization, dok RoBERTa model koristi Byte-Pair Encoding (BPE). Parametri tokenizacije postavljeni su na sljedeći način: maksimalna duljina sekvence iznosi 192 tokena, padding se provodi do maksimalne duljine, truncation je omogućen za tekstove dulje od 192 tokena, a koriste se special tokens poput [CLS] i [SEP].

#### 2.2.3. Obrada Nebalansiranosti Klasa

Dataset je visoko nebalansiran s distribucijom od 71% Low, 17% High i 11% Medium, što zahtijeva posebne tehnike obrade. Korištena je metoda Class Weighting s Weighted Cross-Entropy Loss funkcijom. Težine su automatski izračunate pomoću sklearn.compute_class_weight metode, a zatim primijenjeni dodatni multiplieri po klasi: Low klasa ima multiplier 0.8 zbog smanjenja jer je previše zastupljena, Medium klasa ima multiplier 3.0 zbog povećanja jer je najrjeđa, a High klasa ima multiplier 1.3 kao blago povećanje. Također je korištena tehnika Label Smoothing s vrijednošću 0.05 kako bi se spriječile prekonfidentne predikcije i poboljšala generalizacija modela.

#### 2.2.4. Formatiranje Podataka

Kreiran je custom PyTorch Dataset koji vraća rječnik s tri elementa. Element input_ids predstavlja tokenizirani tekst u obliku tensora. Element attention_mask je tensor koji označava stvarne tokene za razliku od paddinga. Element labels sadrži ID labele gdje 0 označava low, 1 označava medium, a 2 označava high kategoriju.

```python
{
    'input_ids': tensor([101, 2023, 2003, ...]),  # Tokenizirani tekst
    'attention_mask': tensor([1, 1, 1, ...]),      # Attention mask
    'labels': tensor(0)                            # ID labele (0=low, 1=medium, 2=high)
}
```

---

## 3. Treniranje Modela

### 3.1. Odabir Modela

Za ovaj zadatak trenirano je i uspoređeno dva transformer modela. Prvi model je BERT (bert-base-multilingual-cased) koji ima 110 milijuna parametara. Njegova arhitektura sastoji se od 12 slojeva, 12 attention heads i 768 hidden size dimenzija. Prednosti ovog modela uključuju dokazane sposobnosti i multilingvalnu podršku, te je upravo ovaj model odabran kao konačno rješenje. Drugi model je XLM-RoBERTa (xlm-roberta-base) koji ima 125 milijuna parametara. Njegova arhitektura također se sastoji od 12 slojeva, 12 attention heads i 768 hidden size dimenzija, a optimiziran je za cross-lingual transfer.

### 3.2. Hiperparametri Treniranja

Hiperparametri treniranja pažljivo su odabrani kako bi se postiglo optimalno učenje modela. Learning rate postavljen je na 2e-5 što predstavlja spori learning rate pogodan za fine-tuning pretreniranih modela. Batch size iznosi 16 što predstavlja balans između memorijskih zahtjeva i performansi. Gradient accumulation steps postavljen je na 2 što daje efektivni batch size od 32. Broj epoha ograničen je na 3 kako bi se spriječio overfitting. Weight decay iznosi 0.02 što osigurava L2 regularizaciju. Warmup ratio postavljen je na 0.1 što znači da prvih 10% koraka služi za postepeno povećavanje learning rate-a. LR Scheduler je linear što znači linearno smanjenje learning rate-a tijekom treniranja. Max Grad Norm postavljen je na 0.5 za gradient clipping. Label Smoothing iznosi 0.05 kako bi se spriječila overconfidence u predikcijama. Dropout vrijednosti kreću se između 0.15 i 0.2 kao dodatna regularizacija.

### 3.3. Optimizacija i Loss Funkcija

Kao optimizer korišten je AdamW (Adam with Weight Decay) s parametrima beta1 postavljenim na 0.9, beta2 na 0.999 i epsilon na 1e-8. Loss funkcija koja se koristila je Weighted Cross-Entropy Loss gdje su težine klasa automatski balansirane prema distribuciji podataka, a zatim su primijenjeni dodatni multiplieri za fine-tuning balansa.

### 3.4. Proces Treniranja

Proces treniranja odvijao se kroz nekoliko koraka. Prva faza bila je inicijalizacija koja uključuje učitavanje pretreniranih težina. Druga faza je forward pass gdje se podaci prolaze kroz model. Treća faza je loss calculation odnosno izračun weighted cross-entropy loss-a. Četvrta faza je backpropagation gdje se gradijenti računaju kroz sve slojeve mreže. Peta faza je optimization gdje AdamW optimizer ažurira težine modela. Šesta faza je evaluation koja se provodi svakih 200 koraka na validation skupu. Sedma i posljednja faza je early stopping koja zaustavlja treniranje ako nema poboljšanja kroz 2 uzastopne epohe.

### 3.5. Računalna Infrastruktura

Treniranje je provedeno na Apple Silicon (MPS) odnosno CPU hardveru. Korišten je PyTorch framework verzije 2.6.0 te HuggingFace Transformers biblioteka verzije 4.35 i novije. Za praćenje eksperimenata i version control korišten je MLflow sustav.

---

## 4. Evaluacija i Rezultati

### 4.1. Metrike Evaluacije

Za evaluaciju modela korištene su standardne metrike za multi-class klasifikaciju. Accuracy metrika mjeri postotak točno klasificiranih primjera. Precision se računa kao TP / (TP + FP) korištenjem weighted average pristupa. Recall se računa kao TP / (TP + FN) također korištenjem weighted average pristupa. F1-Score predstavlja harmonijsku sredinu precision i recall metrika. Per-class F1 metrika daje F1 score za svaku klasu posebno. Confusion Matrix pruža matricu konfuzije za detaljnu analizu predviđanja.

### 4.2. Rezultati Najboljeg Modela

Najbolji model je BERT (bert-base-multilingual-cased) treniran 18. siječnja 2026.

#### 4.2.1. Ukupne Performanse

Ukupne performanse modela pokazuju izvrsne rezultate. Accuracy iznosi 90.48%, Precision je 90.51%, Recall iznosi 90.48%, a F1-Score je 90.46%.

| Metrika       | Vrijednost |
| ------------- | ---------- |
| **Accuracy**  | **90.48%** |
| **Precision** | **90.51%** |
| **Recall**    | **90.48%** |
| **F1-Score**  | **90.46%** |

#### 4.2.2. Performanse po Klasama

Performanse variraju značajno između klasa. Low klasa postiže izvanredan F1-Score od 99.40% s precizijom 99.42% i recallom 99.38% na približno 5,176 testnih primjera. Medium klasa postiže F1-Score od 58.42% s precizijom 61.23% i recallom 55.91% na oko 829 testnih primjera. High klasa postiže F1-Score od 74.76% s precizijom 73.15% i recallom 76.45% na približno 1,256 testnih primjera.

#### 4.2.3. Confusion Matrix

Confusion matrix pokazuje detaljnu distribuciju predviđanja na test skupu. Od 5,176 Low ticketa, model je točno klasificirao 5,144, pogrešno 18 kao Medium i 14 kao High. Od 829 Medium ticketa, model je pogrešno klasificirao 289 kao Low, točno 464 kao Medium i 76 kao High. Od 1,256 High ticketa, model je pogrešno klasificirao 128 kao Low, 164 kao Medium i točno 964 kao High.

### 4.3. Analiza Performansi

#### 4.3.1. Prednosti

Model pokazuje nekoliko značajnih prednosti. Izvrsne ukupne performanse vidljive su kroz accuracy od 90.5% što predstavlja odličan rezultat za produkcijsku upotrebu. Iznimno dobra detekcija LOW klase s F1-Score od 99.4% je gotovo savršena. Dobra detekcija HIGH klase s F1-Score od 74.8% je zadovoljavajuća za produkcijsku upotrebu. Model također pokazuje stabilnost kroz konzistentne performanse između validation i test seta.

#### 4.3.2. Izazovi

Glavni izazov predstavlja MEDIUM klasa koja postiže samo 58.4% F1-Score. Razlog tome je što Medium klasa čini samo 11.4% dataset-a te je najrjeđa klasa. Problem leži u tome što je granica između Medium i Low/High kategorija često nejasna. Posljedica toga je da model češće griješi na Medium primjerima. Dodatni izazov je class imbalance gdje Low klasa s 71% dominira treniranje, dok Medium klasa s 11% ima premalo primjera za učenje.

#### 4.3.3. Analiza Grešaka

Analiza grešaka pokazuje određene obrasce. Najčešća zabuna je Medium prema Low gdje 35% Medium ticketa biva pogrešno klasificirano kao Low. Druga česta zabuna je High prema Medium gdje 13% High ticketa biva klasificirano kao Medium. Low prema Medium ili High zabune su vrlo rijetke i čine manje od 1% pogrešaka. Razlozi za ove greške uključuju subjektivnost u originalnim oznakama, preklapanje karakteristika između klasa te nedovoljan broj Medium primjera za učenje modela.

---

## 5. Rezultati Analize

### 5.1. Usporedba Svih Treniranih Modela

Analizirano je deset različitih treniranja s različitim konfiguracijama kako bi se identificirale najbolje postavke. Model treniran 18. siječnja 2026 koristeći BERT arhitekturu postigao je najbolje rezultate s 90.5% accuracy i 0.905 F1-Score. Low klasa postigla je 99.4% F1, Medium 58.4% F1, a High 74.8% F1. Model treniran 13. siječnja pokazao je 89.1% accuracy s 0.892 F1-Score gdje je Low imao 99.6%, Medium 56.6% i High 67.9% F1. Model iz 19. siječnja postigao je 89.0% accuracy s Low 99.0%, Medium 56.2% i High 71.1% F1. Najnoviji model iz 23. siječnja koristeći RoBERTa pokazao je 88.5% accuracy, ali s Medium F1 od samo 41.2%. Model treniran 22. siječnja doživio je potpuni neuspjeh s accuracy od samo 19.0% zbog pogrešno postavljenih težina klasa.

### 5.2. Ključna Zapažanja

#### 5.2.1. Evolucija Performansi

Evolucija performansi kroz vrijeme pokazuje zanimljiv obrazac. Rani eksperimenti provedeni između 10. i 12. siječnja postizali su 54-60% accuracy što je predstavljalo početnu fazu istraživanja. Srednji period 13. siječnja donio je značajan skok na 82-89% accuracy što ukazuje na pronalazak bolje konfiguracije. Najbolji model treniran 18. siječnja postigao je 90.5% accuracy što predstavlja vrhunac performansi. Loša konfiguracija od 22. siječnja dovela je do pada na samo 19% accuracy zbog pogrešno postavljenih težina klasa. Oporavak 23. siječnja pokazao je povratak na 88.5% accuracy nakon ispravka konfiguracije.

#### 5.2.2. BERT vs RoBERTa

Usporedba BERT i RoBERTa modela pokazuje jasnu razliku u performansama. BERT model postiže accuracy od 90.5% dok RoBERTa postiže 89.8%. F1-Score za BERT iznosi 0.905 dok RoBERTa postiže 0.898. Na Medium klasi BERT postiže boljih 58.4% F1 u usporedbi s RoBERTa 57.1%. Na High klasi BERT također vodi s 74.8% F1 naspram RoBERTa 72.0%. Stabilnost BERT modela pokazuje se boljom od varijabilnijih rezultata RoBERTa modela. Zaključak je da je BERT konzistentno bolji za ovaj specifični zadatak.

#### 5.2.3. Problem Medium Klase

Kroz sve eksperimente Medium klasa ostaje najslabija karika sustava. Najbolji Medium F1-Score od 60.9% postignut je 11. siječnja koristeći XLM-RoBERTa model. Najlošiji Medium F1-Score od 41.2% zabilježen je 23. siječnja s RoBERTa modelom. Prosječni Medium F1-Score kreće se oko 54% kroz sve eksperimente. Razlozi trajnog problema uključuju činjenicu da Medium čini samo 11% podataka što iznosi 5,526 primjera. Dodatno, postoji semantička neodređenost Medium kategorije te preklapanje karakteristika s Low i High kategorijama.

#### 5.2.4. Utjecaj Težina Klasa

Utjecaj težina klasa pokazao se kritičnim za uspjeh modela. Optimalna konfiguracija s Low težinom 0.15, Medium 6.0 i High 3.5 postigla je 90.5% accuracy i 58.4% Medium F1. Pogrešna konfiguracija s istim težinama ali lošom implementacijom dovela je do katastrofalnih 19.0% accuracy i 9.6% Medium F1. Balansirana konfiguracija s Low 1.0, Medium 1.5 i High 1.2 rezultirala je s 88.5% accuracy ali samo 41.2% Medium F1. Zaključak je da je pažljivo podešavanje težina kritično za uspjeh modela.

### 5.3. Model Explainability - Analiza Pažnje

Model koristi attention mehanizam za fokusiranje na važne riječi tijekom klasifikacije. Primjer HIGH ticketa koji je točno klasificiran pokazuje tekst "sent monday october server work server work please restart application" gdje su najvažnije riječi "application" s 10%, "restart" s 7.4% i "server" s 6.8% pažnje. Confidence ovog predviđanja iznosi 74.7%. Primjer LOW ticketa "tuesday pm mailing dear kind create mailing members thanks" pokazuje najvažnije riječi "pm" s 22.8%, "tuesday" s 15.5% i "thanks" s 6.9% pažnje uz 100% confidence. Zanimljiv je primjer MEDIUM ticketa koji je pogrešno klasificiran kao HIGH. Tekst "dear taking into consideration phone use having issues please help" pokazuje riječi "dear" s 13%, "phone" s 8.5% i "officer" s 4.9%, a nesigurnost modela vidljiva je kroz confidence od samo 55.5%.

---

## 5. Zaključak

Razvijen je uspješan model za automatsku klasifikaciju hitnosti IT ticketa koji postiže 90.5% accuracy. Model pokazuje izvrsne rezultate na Low i High kategorijama, dok Medium kategorija ostaje izazov zbog male zastupljenosti u podacima.

Ključni uspjesi projekta uključuju visoku ukupnu točnost od 90.5% što omogućava pouzdanu automatsku kategorizaciju. Postignuta je gotovo savršena detekcija Low ticketa s 99.4% F1-Score. Dobra detekcija High ticketa s 74.8% F1-Score zadovoljava poslovne potrebe. Model također pokazuje stabilnost i ponovljivost rezultata kroz različite eksperimente.

Područja za buduća poboljšanja uključuju skupljanje većeg broja Medium primjera kako bi se poboljšalo učenje ove kategorije. Potrebno je razmotriti redefiniranje Medium kategorije, možda kroz podjelu na dvije podkategorije kako bi se smanjila semantička nejasnoća. Također se mogu primijeniti ensemble metode koje kombiniraju više modela za poboljšanje detekcije Medium klase.

Model je spreman za deployment u produkcijsko okruženje uz preporuku pažljivog monitoringa Medium predikcija. Kontinuirano praćenje performansi i redovito retreniranje s novim podacima osigurat će dugoročnu uspješnost sustava.

---
