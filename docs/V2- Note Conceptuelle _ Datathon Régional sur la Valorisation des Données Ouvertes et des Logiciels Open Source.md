# **Note Conceptuelle : Datathon Régional \- Évaluation socioculturelle des grands modèles de langage** 

---

**Organisateur principal :** Inria Chile  
**Partenaires :** Ambassades de France en Amérique latine, Ministère de l’Europe et des Affaires Étrangères, Réseau d’institutions académiques latino-américaines.   
**Contexte :** Assises de la Science Ouverte organisées à Montevideo, en Uruguay, du 28 au 30 octobre 2026\. 

---

##  

## **1\. Contexte et Justification**

### **1.1. Cadre général**

Les Assises de la Science Ouverte, organisées à Montevideo en octobre 2026, constituent un événement majeur pour promouvoir les principes de transparence, de partage et de collaboration dans la recherche scientifique. 

Dans ce cadre, Inria Chile propose d’organiser un **datathon régional**, en collaboration avec les Ambassades de France de la région et un réseau d’universités et d’instituts de recherche latino-américains. 

Inria Chile est l’opérateur du Centre Binational Franco-Chilien d’Intelligence Artificielle et l’une des priorités de ce Centre, en lien avec les stratégies d’IA de la France et des pays de la région en particulier du Chili, est l’évaluation de l’IA, pour le développement et l’adoption d’une IA responsable, de confiance, et fiable. 

Actuellement, les grands modèles de langage les plus connus sont massivement utilisés en Amérique latine. À titre d’exemple, on peut citer le Chili, où [plus de 76% de la population adulte](https://portalinnova.cl/76-de-los-chilenos-ya-usa-inteligencia-artificial-y-un-40-la-utiliza-para-aprender-o-estudiar/) utilise l’IA générative- l’un des taux les plus haut au monde, et le Brésil, 3ème marché du monde pour OpenAI et 2ème pays dont la croissance est la plus rapide pour les applications payantes (+161% [pour OpenaAI](https://cdn.openai.com/pdf/7ef17d82-96bf-4dd1-9df2-228f7f377a29/the-state-of-enterprise-ai_2025-report.pdf) entre novembre 2024 et novembre 2025, derrière l’Australie et devant la France). 

Cependant, les données utilisées pour alimenter ces grands modèles de langage proviennent majoritairement des pays du Nord et sont en anglais, ce qui peut produire du contenu biaisé ou des lacunes de connaissances lorsque ces modèles sont appliqués à des contextes culturels du Sud, du fait des données présentes (ou sous représentées) pour l'entraînement de ces modèles. 

### **1.2. Le datathon régional** 

Fin 2025, Inria Chile, avec des chercheurs d’Inria en France et des partenaires académiques chiliens, ont lancé le projet *LLACA: LLM Assimilation of Cultural Aspects* pour construire un système avancé qui permette d’évaluer et améliorer les connaissances socioculturelles des modèles de langage, en se concentrant initialement sur l'Amérique latine. 

Dans ce cadre, les scientifiques ont développé l’outil LatamQA, grâce à une méthodologie reproductible qui combine Wikidata, Wikipedia et l’expertise de scientifiques en sciences humaines et sociales pour produire des questions d’évaluation culturelle de haute qualité en espagnol (variantes latino-américaine et ibérique), portugais brésilien et anglais.

La version actuelle de LatamQA, [dont les premiers résultats ont déjà été publiés](https://inria.hal.science/hal-05510068/document%20), présente une couverture inégale selon les pays et dépend largement de la couverture de Wikipedia et Wikidata, qui tend à être plus riche pour les pays disposant d’une infrastructure éditoriale plus développée.

**Le datathon vise à corriger cette asymétrie grâce à des contributions directes d’experts locaux : étudiants, chercheurs, universitaires et professionnels latino-américains qui connaissent leur contexte culturel de première main et de réaliser un diagnostic profond des biais culturels, linguistiques, historiques, de genre et de minorités des grands modèles de langage.**  

Pour cela, les participants seront invités à produire des questions et des réponses pour évaluer les modèles de langage, en fonction de leurs pays, cultures, langues et régions d’origine. 

Chaque équipe créera un ensemble de questions suivant le modèle des questions à choix multiples (une seule réponse correcte parmi quatre). Par exemple, une question pourrait être :

| Campo | Valor |
| :---- | ---- |
| `Pregunta:` | ¿Qué es el milcao? |
| `Opción A:` | Un batido a base de chocolate.|
| `Opción B:` | Un pájaro de la Araucanía. |
| `Opción C:` | Una herramienta para ordeñar cabras del pueblo mapuche. |
| `Opción D:` | Un pan de papas típico de Chiloé. |
| `Respuesta correcta:` | D. |
| `Referencia Wikipedia:` | [Q1041870](https://www.wikidata.org/wiki/Q1041870) | 


Chaque équipe participante fournira une liste de questions visant à capturer des connaissances spécifiques à leur région, pays ou culture, où une seule des options proposées est la bonne réponse. Chaque question devra être formulée dans la langue locale (espagnol ou portugais) et en anglais.

Inria Chile fournira une plateforme web pour faciliter et coordonner la participation des équipes.

La compétition sera réalisée sur une journée à l’issue de laquelle une équipe gagnante sera désignée par pays participant, et parmi ses équipes, l’équipe gagnante de la Datathon Régionale. L’équipe vainqueur sera annoncée lors de l’inauguration des Assises de Montevideo, lors desquelles une vidéo récap du Datathon sera diffusée.  

**1.3. Objectifs principaux**

* **Sensibiliser** les étudiants et jeunes chercheurs d’Amérique latine aux enjeux de la science ouverte et du logiciel open source et aux enjeux et risques liés à l’utilisation des grands modèles de langage actuellement.   
* **Renforcer** la cohésion culturelle et l’exploration interdisciplinaire (sciences de l’informatique ; ingénierie ; sciences humaines et sociales).  
* **Encourager** la collaboration régionale et la création de projets concrets répondant à des défis locaux ou globaux.  
* **Valoriser** les meilleurs apports à LatamQA et l’engagement des participants. 

### **1.4. Public visé** 

La jeunesse latinoaméricaine, et en particulier, les étudiants latinoaméricains de niveau licence, master et doctorat en sciences humaines et sociales, sciences de l’informatique, etc. des pays participants. 

**1.5. Conformation des équipes** 

* Les partenaires participants dans chaque équipe respective pourront limiter le nombre d’équipes participantes, avec un minimum de 2 équipes.   
* Chaque équipe doit être composée de 3 à 6 membres, dont au moins un ayant une formation technique en traitement du langage naturel ou en science des données, et au moins un ayant une formation en sciences humaines et sociales. Les équipes devront être interdisciplinaires.   
* La diversité de manière générale sera fortement valorisée, notamment la diversité régionale, de genre, d’âge, de cultures sera particulièrement valorisée avec l’objectif de refléter l'hétérogénéité interne de chaque État.  
* Tous les membres doivent être ressortissants ou résidents permanents du pays qu'ils représentent. 

**1.6. Évaluation et prix** 

1. Évaluation 

Les participants seront invités à produire un dataset de questions et de réponses pour évaluer les modèles de langage, en fonction de leurs pays, cultures, langues et régions d’origine. 

Les réponses des équipes seront évaluées par Inria Chile. Pour cela, Inria Chile utilisera un panel de grands modèles de langage (LLMs) open source. L’évaluation de chaque équipe dépendra du nombre de questions proposées par l’équipe qui parviennent à tromper le panel de LLMs.

À la suite de l’évaluation : 

- 1 équipe gagnante sera choisie par pays et annoncée le jour même de la datathon   
- Parmi ces équipes, l’équipe gagnante régionale sera annoncée durant les Assises.   
    
2. **Prix** 

Les équipes gagnantes par pays recevront un panier garni. 

L’équipe gagnante recevra 2000 euros à répartir équitablement entre ses membres. Ses membres seront également cités dans la publication de LatamQA v2 d’Inria Chile. 

---

## **2\. Format et partenaires** 

### **2.1. Format de l’événement**

* **Durée :** une journée (détails encore à préciser)   
* **Date :** le samedi 3 octobre 2026  
* **Modalité :**   
  * La datathon sera réalisée à la même date et aux mêmes horaires dans tous les pays participants.   
  * Pour renforcer la cohésion des équipes, l’organisateur proposera aux partenaires nationaux la réalisation de l’événement de préférence en présentiel. Cependant, si le partenaire national n’est pas en mesure d’organiser l’événement de manière présentiel, les équipes du pays en question pourront participer en ligne.   
* **Langues :** espagnol dans ses différentes variantes et portugais brésilien.   
* **Outils numériques :**    
  * Pour la communication : [Discord](https://discord.com/) ou [Matrix](https://matrix.org/foundation/about/) (outil de communication sécurisé et décentralisé), mis à disposition par l’organisateur.   
  * Plateforme : Outil open source de collaboration pour créer des ensembles de données de haute qualité. 

### **2.2. Partenaires de l’événement** 

* **Inria Chile :** Coordination générale, mise en place du format et des outils numériques, communication, évaluation et sélection.  
* **Ambassades de France :** Mobilisation des réseaux académiques, soutien logistique, communicationnel et financier.  
* **Institutions académiques :** Organisation locale, soutien logistique, et communicationnel. 

## **4\. Aspects Organisationnels et Financiers**

### **4.1. Calendrier Prévisionnel**

| Étape | Date préliminaire | Responsables | Commentaires additionnels |
| ----- | ----- | ----- | ----- |
| **Mobilisation des postes des pays concernés**  | À partir de la semaine du 15 juin  | Ambassades de France en Argentine et Uruguay  | Définir conjointement en amont la liste des pays concernés :  Argentine Brésil Chili  Mexique Uruguay À compléter si besoin  |
| **Coordination avec les postes et les partenaires**  | Juillet-Septembre  Réunion avec les partenaires début septembre  (session d’information)  | Ambassades et Inria Chile si besoin  | La liste des universités participantes doit être claire au plus tard fin juillet.  |
| **Lancement de l’appel à participation** | 1 septembre 2026 | Inria Chile  | Avec le soutien des partenaires institutionnels  |
| **Annonce de sélection et/ou conformation des équipes** | 21 septembre 2026 | Inria Chile  |  |
| **Datathon régional** | 3 octobre 2026 | Inria Chile \+ partenaires |  |
| **Annonce de l’équipe gagnante durant les Assises à Montevideo et remise (virtuelle) du prix**   | 28 octobre | Inria Chile et Ambassades de France en Argentine et Uruguay  | Cette option est viable seulement si les participants peuvent se connecter en ligne.  |

### **4.2. Communication** 

* **Cibles :** Étudiants, chercheurs, institutions académiques, médias spécialisés.  
* **Canaux :**  
  * Réseaux sociaux (Twitter, LinkedIn, Instagram, Youtube).  
  * Newsletters Inria Chile, universités et ambassades partenaires.  
  * Médias locaux et régionaux.   
* **Objets communicationnels :** landing page ; teaser de promotion et vidéo post événement ; logo et branding ; guide et règles de participation ; kit de communication ; communiqué et gestion de presse ; etc. 

### **4.3. Budget prévisionnel** 

Inria Chile apportera des moyens humains au pilotage et à l’organisation de la datathon notamment pour :  définir les conditions de participation des équipes, évaluer les équipes, réaliser le support technique de l’événement, préparer et réaliser les aspects communicationnels ainsi que coordonner les différents aspects avec les institutions participantes et les postes dans les différents pays. L’estimation de ces coûts en ressources humaines est en cours.  

En termes de communication, l’estimation a minima des coûts est de : 

- Capsules teaser et post événement (estimé à 3000 euros- devis à réaliser)   
- Logo, branding, support communicationnels digitaux (3500 euros- devis à réaliser))   
- Landing page (970 euros- devis à réaliser)  

En soutien à son équipe, Inria Chile envisage de recruter un stagiaire pour une durée de 4 mois à un coût estimé de 1400 euros sur la période. 

L’estimation budgétaire pécuniaire est d’environ 9000 euros. 

\*Ce budget ne prend pas en compte les supports matériels de communication (goodies, posters et autres prints, tee-shirts, etc.) qui dépendent du nombre d’institutions et de pays participants ni le paiement de photographe ou vidéaste sur place. Chaque institution participante devra envoyer du matériel audiovisuel de qualité à l’organisateur pour la réalisation de produits communicationnels post événement. 

\*\*Ce budget ne prend par ailleurs pas en compte l’hébergement sur site de la datathon, ni l’alimentation, le transport, la connexion internet et le matériel informatique que les participants pourraient nécessiter sur place.  

\*\*\*Ce budget ne prend pas en compte de prix pour les gagnants. 

---

## 

## **Annexe. Quelles questions et réponses pour les participants au Datathon ?** 

Le but de la datathon pour les équipes est de rédiger et soumettre aux LLMs des questions et des réponses liées à des aspects culturels latinoaméricains. 

1. Type de questions : 

Chaque question doit respecter les critères suivants :

* **Type** : Question à choix multiples (QCM) avec 4 options (1 bonne réponse \+ 3 distracteurs plausibles).  
* **Langue** :  
  * Rédigée dans la langue principale du pays de l’équipe (espagnol local ou portugais brésilien).  
  * Traduction en anglais obligatoire pour chaque question.  
* **Lien avec Wikidata** :  
  * Chaque question doit être liée à une ou plusieurs entités Wikidata (obligatoire sauf exception).  
  * Si impossible, une justification doit être fournie dans le champ "procedencia cultural" (origine culturelle).  
* **Métadonnées obligatoires** :  
  * Pays d’origine.  
  * Région subnationale (si applicable, ex. : État, province).  
  * Dimension culturelle (ex. : histoire, politique, art, musique, gastronomie, sport, peuples autochtones, langue, religion, géographie, etc.).  
  * Note contextuelle en espagnol ou portugais (pour expliquer le contexte culturel).  
2. Limites par équipe :   
* **Nombre de questions** :  
  * Chaque équipe peut soumettre jusqu’à 60 questions pendant la phase de construction.  
  * Seules les 30 meilleures questions (selon leur score individuel) seront comptabilisées pour le classement final.  
  * Objectif : Encourager l’exploration sans pénaliser les essais infructueux.  
* **Détection des doublons** :  
  Un système automatique repère les questions semantiquement similaires (seuil : \> 0,92 en embeddings multilingues).  
  * En cas de doublon confirmé entre équipes, la première version soumise est conservée.  
3. Critères d’exclusion :

Une question sera éliminée du score final si elle :

1. Dépend de données éphémères ou non vérifiables (rumeurs, opinions, évènements sans documentation publique).  
2. Est une trivia obscure sans contenu culturel significatif (évalué par le comité).  
3. Ne dépasse pas le seuil de validité dans le pays concerné (voir section 5).  
4. Contient des erreurs factuelles, des stéréotypes nuisibles ou un contenu discriminatoire.  
5. Viole le droit d’auteur ou la licence MIT du dépôt LatamQA.