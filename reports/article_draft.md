# Détection du langage « Nous contre Eux » sur les réseaux sociaux : un pipeline NLP modulaire pour l'analyse de l'altérisation

---

## Page de garde

| | |
|---|---|
| **Étudiant** | Julian Ray-Constanty |
| **Titre du rapport** | Détection du langage d'altérisation sur les réseaux sociaux : pipeline NLP multi-étapes |
| **Type d'expérience** | Stage de fin de formation |
| **Dates du stage** | 14 avril 2026 – 6 juin 2026 |
| **Établissement d'accueil** | [NOM DE L'ORGANISME D'ACCUEIL] |
| **Diplôme préparé** | BUT Informatique |
| **Établissement universitaire** | IUT du Limousin – Université de Limoges |
| **Tuteur entreprise** | [NOM DU TUTEUR ENTREPRISE] |
| **Tuteur pédagogique** | [NOM DU TUTEUR PÉDAGOGIQUE] |

---

## Remerciements

Je tiens à remercier [NOM ET FONCTION DU TUTEUR ENTREPRISE] pour son encadrement tout au long de ce stage, ainsi que [NOM DU TUTEUR PÉDAGOGIQUE], tuteur pédagogique à l'IUT, pour ses retours réguliers sur l'avancement du projet. Je remercie également l'équipe de [L'ORGANISME D'ACCUEIL] pour l'accueil et les échanges qui ont nourri ce travail.

---

## Sommaire

1. Introduction
2. Présentation de l'organisme d'accueil
   - 2.1 L'organisation et son secteur
   - 2.2 L'équipe et le contexte d'intégration
   - 2.3 Ma mission : périmètre et enjeux
3. Données et méthodologie
   - 3.0 Organisation et gestion du projet
   - 3.1 Sources de données
   - 3.2 Statistiques du corpus
   - 3.3 Prétraitement
   - 3.4 Étape 1 : Balisage des pronoms
   - 3.5 Étape 2 : Scoring de toxicité
   - 3.6 Étape 3 : Classification des émotions
   - 3.7 Étape 4 : Détecteur rule-based d'altérisation
   - 3.8 Étape 5 : Classificateur supervisé
   - 3.9 Étape 6 : Modélisation thématique (BERTopic)
4. Résultats
   - 4.1 Distribution des pronoms
   - 4.2 Toxicité
   - 4.3 Émotions
   - 4.4 Détection de l'altérisation
   - 4.5 Performance du classificateur
   - 4.6 Modélisation thématique
   - 4.7 Comparaison par plateforme
5. Discussion
6. Conclusion
7. Glossaire
8. Bibliographie
9. Annexes

---

## 1. Introduction

En 2023, Meta a supprimé plus de 1,3 milliard de faux comptes en un seul trimestre, tout en admettant que la détection des discours de haine *implicites* reste un problème ouvert (Meta Transparency Report, 2023). Ce chiffre illustre l'ampleur du défi : modérer du contenu à l'échelle de milliards d'utilisateurs est un problème aussi technique que linguistique.

C'est dans ce contexte que j'ai effectué mon stage de fin de BUT Informatique au sein de [L'ORGANISME D'ACCUEIL], structure spécialisée dans [DOMAINE DE L'ORGANISME]. L'équipe travaillait sur la problématique de la détection automatique des contenus polarisants sur les réseaux sociaux, et avait besoin d'un pipeline NLP reproductible capable d'analyser de larges corpus de textes en anglais issus de plateformes comme Reddit ou Twitter. Mon profil correspondait à ce besoin : formation BUT Informatique avec des modules de programmation Python, traitement de données et algorithmique, combinés à un intérêt personnel pour les sujets liés à la linguistique computationnelle et à la cybersécurité des contenus.

**Problématique :** Comment construire un pipeline NLP modulaire permettant de détecter et de caractériser automatiquement le langage d'altérisation (c'est-à-dire la construction rhétorique d'un « eux » menaçant face à un « nous » légitime) dans des contenus de réseaux sociaux à grande échelle ?

Ce rapport présente les choix techniques effectués, les résultats obtenus et les limites du travail. Dans un premier temps, je présente l'organisme d'accueil et le contexte du stage. Je décris ensuite les données utilisées et la méthodologie du pipeline en six étapes. Je présente les résultats obtenus sur 134 459 posts, avant d'ouvrir une discussion critique sur les limites et les pistes d'amélioration.

---

## 2. Présentation de l'organisme d'accueil

### 2.1 L'organisation et son secteur

[À REMPLIR : Secteur d'activité, statut juridique, taille, historique, produits/services principaux, positionnement sur le marché. 1 à 2 paragraphes.]

[Exemple de structure : « [L'ORGANISME] est une structure de [type : laboratoire / start-up / association / entreprise] fondée en [année], dont l'activité principale est [domaine]. Elle compte [n] collaborateurs et intervient principalement dans le secteur [secteur]. Son offre de services repose sur [produits/missions principales]. »]

### 2.2 L'équipe et le contexte d'intégration

[À REMPLIR : Décrire l'équipe dans laquelle vous étiez intégré : composition, rôles, organisation. Comment se prenaient les décisions ? Quels outils de collaboration ? Quelle ambiance de travail ?]

[Préciser les interactions : « J'interagissais principalement avec [profils : chef de projet, data scientist, développeur…]. Les méthodes de travail étaient [agiles / en cycle en V / informelles…]. »]

### 2.3 Mission et périmètre du stage

La mission qui m'a été confiée consistait à concevoir et implémenter un pipeline NLP complet pour analyser le discours d'altérisation sur les réseaux sociaux. L'existant se limitait à des scripts de collecte partiels et à quelques expérimentations isolées. L'objectif était de produire un système reproductible, modulaire, et documenté, capable de traiter des dizaines de milliers de posts et de restituer les résultats dans un tableau de bord interactif.

Le projet couvrait l'ensemble de la chaîne de traitement : collecte des données, nettoyage, enrichissement par des modèles pré-entraînés, détection par règles, classification supervisée, modélisation thématique et visualisation. Un rapport technique et une présentation finale étaient attendus en livrable.

---

## 3. Données et méthodologie

### 3.0 Organisation et gestion du projet

Le stage s'est déroulé sur 8 semaines (14 avril – 6 juin 2026), organisées selon un découpage itératif en sprints hebdomadaires, inspiré des pratiques agiles. Chaque semaine correspondait à une étape fonctionnelle du pipeline, avec un livrable intermédiaire clairement défini (fichier CSV enrichi, modèle entraîné, figure exportée). Ce découpage permettait de valider chaque composant avant de passer au suivant, et de détecter les problèmes de données tôt.

| Semaine | Livrable |
|---|---|
| S1 | Corpus nettoyé (134 459 posts) |
| S2 | Dataset enrichi (toxicité + émotions) |
| S3 | Dataset avec scores d'altérisation rule-based |
| S4 | Classificateur ML (F1=0,990) + dataset classifié |
| S5 | Modélisation BERTopic + dataset final |
| S6 | Comparaison plateforme + figures temporelles |
| S7 | Dashboard Streamlit complet |
| S8 | Rapport + présentation |

*Table 0 : Découpage du projet par semaine.*

Les outils de suivi utilisés étaient simples et autonomes : un fichier de planning hebdomadaire, un dépôt Git local pour le versioning du code, et un nommage systématique des fichiers intermédiaires (ex. `dataset_othering.csv`, `dataset_classified.csv`) pour garantir la traçabilité à chaque étape. Le travail était réalisé en autonomie avec des points réguliers avec le tuteur entreprise.

Le pipeline suit six étapes séquentielles. Chaque étape ajoute des colonnes au même fichier CSV sans modifier les colonnes précédentes, ce qui garantit la traçabilité et permet de rejouer une étape indépendamment des autres.

```
dataset_clean.csv
        │
        ▼
[Étape 1] Balisage pronoms ──► dataset_pronouns.csv
        │
        ▼
[Étape 2+3] Toxicité + Émotions ──► dataset_enriched.csv
        │
        ▼
[Étape 4] Détecteur rule-based ──► dataset_othering.csv
        │
        ▼
[Étape 5] Classificateur ML ──► dataset_classified.csv
        │
        ▼
[Étape 6] BERTopic ──► dataset_final.csv
```

*Figure 1 : Architecture du pipeline en six étapes.*

### 3.1 Sources de données

Le corpus combine deux sources publiques. La première est le **UC Berkeley Measuring Hate Speech corpus** (Kennedy et al., 2020), chargé depuis HuggingFace (`ucberkeley-dlab/measuring-hate-speech`, split train). Il contient des posts issus principalement de Twitter et Reddit, annotés par des travailleurs crowdsource sur plusieurs dimensions de la nocivité. Nous utilisons le champ `annotator_severity` comme score de sévérité.

La seconde source est l'**archive Pushshift Reddit** (Baumgartner et al., 2020), accessible via HuggingFace (`fddemarco/pushshift-reddit`). Cinq subreddits ont été sélectionnés pour leur contenu politique : r/politics, r/worldnews, r/conspiracy, r/europe et r/immigration.

Le choix de ces sources répond à un double besoin : la présence de contenu labelisé pour l'entraînement supervisé (corpus Berkeley) et un ancrage dans des communautés politiques réelles (Reddit).

### 3.2 Statistiques du corpus

Après suppression des posts de moins de 20 caractères et déduplication, le corpus final compte **134 459 posts**. Parmi eux :

| Source | Posts | Part |
|---|---|---|
| Hate speech corpus (Berkeley) | 133 170 | 99,04 % |
| Reddit (5 subreddits) | 1 289 | 0,96 % |

Au sein de Reddit : r/politics contribue 782 posts, r/worldnews 407, r/conspiracy 87, r/europe 13.

Ce déséquilibre est important à garder en tête : le corpus Berkeley a été collecté pour surreprésenter les contenus nocifs. Les statistiques agrégées (taux de toxicité, taux d'altérisation) reflètent donc davantage la composition du corpus que l'état général des réseaux sociaux.

### 3.3 Prétraitement

Chaque post passe par une fonction de nettoyage qui : convertit le texte en minuscules, supprime les URLs, les mentions `@`, et normalise les caractères spéciaux en conservant la ponctuation de base. Le texte nettoyé est stocké dans la colonne `clean_text`, qui sert d'entrée à toutes les étapes suivantes.

### 3.4 Étape 1 : Balisage des pronoms

Deux familles de pronoms sont identifiées. Les marqueurs **WE** couvrent les formes de première personne du pluriel en anglais : *we, us, our, ours, ourselves*. Les marqueurs **THEM** couvrent la troisième personne du pluriel et plusieurs expressions multi-mots à valeur dépréciative : *they, them, their, theirs, those people, these people, people like them*.

Pour chaque post, on enregistre des indicateurs binaires de présence, le comptage brut, et une catégorie `pronoun_type` : `we_only`, `them_only`, `both`, ou `none`. La liste THEM est délibérément plus large que la grammaire stricte : des formules comme *« those people »* ou *« people like them »* portent un cadrage excluant que les pronoms seuls ne captent pas.

### 3.5 Étape 2 : Scoring de toxicité

Le modèle **Detoxify** (`original`) est appliqué sur l'ensemble du corpus par lots de 32. Il produit cinq scores flottants entre 0 et 1 : `toxicity`, `severe_toxicity`, `identity_attack`, `insult`, `threat`. Le seuil 0,7 est retenu comme seuil de haute toxicité dans les analyses suivantes. Le traitement par lots permet de parcourir les 134 459 posts en quelques heures sur CPU.

### 3.6 Étape 3 : Classification des émotions

Le modèle **GoEmotions** (`monologg/bert-base-cased-goemotions-original`) est utilisé via l'API HuggingFace pipeline avec `top_k=1`. Il assigne à chaque post une émotion dominante parmi 27 catégories et un score de confiance. Les entrées sont tronquées à 512 caractères. Cette troncature affecte une minorité de posts mais peut couper les segments les plus émotionnellement chargés des textes longs ; c'est une limite acceptable.

### 3.7 Étape 4 : Détecteur rule-based d'altérisation

Le cœur de la détection est un dictionnaire de 33 patterns lexicaux répartis en quatre catégories :

| Catégorie | Exemple de patterns |
|---|---|
| **Métaphores de menace** | *invasion, flood, swarm, horde, plague, wave of* |
| **Exclusion morale** | *go back, don't belong, not like us, their kind, no place here* |
| **Généralisation** | *they always, all of them, none of them ever, these people always* |
| **Cadrage menaçant** | *replace us, great replacement, taking over, destroying our way* |

*Table 1 : Catégories et exemples de patterns du détecteur rule-based.*

L'`othering_score` d'un post est le nombre de catégories qui correspondent (0 à 3), et `has_othering` est vrai dès qu'une catégorie correspond. Chaque pattern correspondant est conservé dans le champ `matched_patterns` pour audit. Le dictionnaire est volontairement conservateur : mieux vaut manquer certains cas que d'introduire du bruit dans le dataset d'entraînement.

### 3.8 Étape 5 : Classificateur supervisé

Pour tester si le signal d'altérisation peut être appris au-delà des patterns explicites, quatre classificateurs sont entraînés avec `has_othering` comme label. Côté features, deux représentations sont comparées : TF-IDF (`TfidfVectorizer`, max 10 000 features) et embeddings de phrases (`all-MiniLM-L6-v2`, lots de 64). Côté modèles, on compare `LogisticRegression(max_iter=1000)` et `LinearSVC(kernel="linear", probability=True)`.

Le split entraînement/test est 80/20 avec `random_state=42`. L'évaluation porte sur la précision, le rappel, le F1 et la matrice de confusion. Une analyse d'erreurs sur le meilleur modèle complète l'évaluation.

### 3.9 Étape 6 : Modélisation thématique (BERTopic)

BERTopic (Grootendorst, 2022) est appliqué sur le corpus nettoyé complet, avec les embeddings `all-MiniLM-L6-v2`, `min_topic_size=50`, et sélection automatique du nombre de topics. Les embeddings et projections UMAP sont mis en cache pour éviter de les recalculer à chaque exécution. Après entraînement, les topics sont examinés manuellement et des noms lisibles leur sont assignés. Les résultats sont croisés avec la toxicité, l'émotion dominante et le taux d'altérisation par topic.

J'ai préféré BERTopic à LDA parce que BERTopic s'appuie sur des embeddings contextuels et produit des topics plus interprétables sur des textes courts et bruités. LDA, basée sur la fréquence des mots, aurait donné des résultats plus bruités sur la nature fragmentée des posts de réseaux sociaux.

---

## 4. Résultats

### 4.1 Distribution des pronoms

Sur les 134 459 posts :

| Type | Posts | Part |
|---|---|---|
| Aucun pronom (none) | 76 127 | 56,6 % |
| THEM uniquement (them_only) | 33 889 | 25,2 % |
| WE uniquement (we_only) | 24 203 | 18,0 % |
| Les deux (both) | 9 009 | 6,7 % |

La majorité des posts n'utilisent ni marqueur WE ni THEM. La présence simultanée des deux familles (6,7 %) est le signal le plus fort pour l'altérisation.

### 4.2 Toxicité

Le score de toxicité moyen est **0,595** et 56,0 % des posts dépassent le seuil 0,7. Ces chiffres élevés s'expliquent principalement par la surreprésentation de contenus nocifs dans le corpus Berkeley. Ils ne sont pas généralisables au contenu moyen des réseaux sociaux.

### 4.3 Émotions

| Émotion | Part |
|---|---|
| Neutral | 35,5 % |
| Anger | 18,2 % |
| Annoyance | 9,9 % |
| Curiosity | 5,6 % |
| Admiration | 4,8 % |

La neutralité domine même dans un corpus de haine : beaucoup de posts sont des constats ou des descriptions. Dans les posts avec altérisation, la colère et le dégoût sont surreprésentés par rapport au corpus global, mais sans rupture catégorique.

### 4.4 Détection de l'altérisation

Le détecteur rule-based a identifié **7 115 posts (5,3 %)** comme contenant de l'altérisation. La ventilation par type de pronoms révèle un gradient très net :

| Type de pronoms | Taux d'altérisation |
|---|---|
| both (we + them) | 22,2 % |
| them_only | 12,3 % |
| we_only | 6,9 % |
| none | 1,2 % |

*Table 2 : Taux d'altérisation par type de pronoms.*

Les posts avec marqueurs THEM sont environ 10 fois plus susceptibles de contenir de l'altérisation que les posts sans aucun pronom. L'ajout des marqueurs WE double encore ce taux. Le balisage des pronoms constitue donc un signal de surface bon marché et efficace pour prioriser les posts à analyser.

Parmi les 7 115 posts flagués, **5 305 (74,6 %)** dépassent aussi le seuil de toxicité 0,7, contre 56,0 % dans le corpus global. L'altérisation et la toxicité sont corrélées, mais pas redondantes : environ un quart des posts d'altérisation ont une toxicité modérée, ce qui correspond à l'exclusion polie, rhétoriquement construite mais sans insulte directe.

### 4.5 Performance du classificateur

Les résultats des deux modèles TF-IDF sur le jeu de test (26 892 posts) :

| Modèle | Features | Précision | Rappel | F1 |
|---|---|---|---|---|
| LogisticRegression | TF-IDF | 0,9428 | 0,9726 | 0,9575 |
| LinearSVC | TF-IDF | 0,9943 | 0,9852 | **0,9898** |

*Table 3 : Performance des classificateurs sur le jeu de test.*

Le LinearSVC TF-IDF est le meilleur modèle avec un F1 de **0,990**. La matrice de confusion confirme la fiabilité : sur 26 892 posts, seulement 29 erreurs (8 faux positifs, 21 faux négatifs).

L'analyse d'erreurs révèle deux modes d'échec. Les faux positifs viennent surtout de posts qui utilisent *invasion* ou *flooding* au sens littéral (catastrophes naturelles, reportages militaires). Les faux négatifs, eux, concernent le langage codé, l'ironie et l'altérisation implicite que le dictionnaire ne couvre pas.

L'accord très proche entre le détecteur rule-based (5,3 %) et le classificateur (5,29 %) indique que le modèle a bien appris les patterns, mais aussi qu'il mémorise probablement le dictionnaire plutôt que de généraliser à l'altérisation au sens large.

### 4.6 Modélisation thématique

BERTopic a identifié **1 334 topics** sur le corpus complet. Les plus grands clusters révèlent les thématiques où le contenu haineux se concentre :

| Topic | Posts | Taux altérisation | Toxicité moy |
|---|---|---|---|
| Injures raciales et abus | ~2 300 | élevé | élevée |
| Débat sur l'avortement | ~1 194 | modéré | modérée |
| Iran / géopolitique nucléaire | ~1 080 | faible | faible |
| Homophobie | ~961 | élevé | élevée |
| Antisémitisme | ~944 | élevé | élevée |

*Table 4 : Principaux topics et profil d'altérisation (top 5 par taille).*
*Données complètes disponibles dans le dashboard interactif (reports/presentation.html).*

Le nombre élevé de topics (1 334) est une limite du paramètre `nr_topics="auto"` sur un corpus aussi large et diversifié. Il produit une décomposition très fine dont une partie relève du bruit. Une prochaine itération bénéficierait d'un nombre de topics fixé ou d'un regroupement hiérarchique post-hoc.

### 4.7 Comparaison par plateforme

Les subreddits Reddit présentent un profil très différent du corpus Berkeley :

| Subreddit | Posts | Toxicité moy | Taux altérisation | Émotion dom. | Pattern dominant |
|---|---|---|---|---|---|
| r/politics | 782 | 0,061 | 0,90 % | neutral | get_out |
| r/worldnews | 407 | 0,047 | 0,49 % | neutral | animals |
| r/conspiracy | 87 | 0,050 | 0,00 % | neutral | n/a |
| r/europe | 13 | 0,002 | 0,00 % | neutral | n/a |

*Table 5 : Comparaison des subreddits Reddit.*

La toxicité moyenne sur Reddit (0,06) est dix fois inférieure à celle du corpus Berkeley (0,60). Le taux d'altérisation de 0,90 % sur r/politics, bien que faible en valeur absolue, est cohérent avec les travaux sur la polarisation politique en ligne (Davidson et al., 2017). Les taux nuls sur r/conspiracy et r/europe sont des artefacts de taille d'échantillon, non des absences réelles de ce type de contenu dans ces communautés.

---

## 5. Discussion

**La structure pronominale comme signal de surface efficace.** Le gradient de 1,2 % à 22,2 % selon le type de pronoms est le résultat le plus robuste de ce travail. Un post contenant à la fois des marqueurs WE et THEM a 18 fois plus de chances de contenir de l'altérisation qu'un post sans aucun pronom. Ce signal est rapide à calculer, ne nécessite aucune inférence de modèle, et pourrait servir de premier filtre dans un système de modération à grande échelle.

**Complémentarité et non-redondance entre altérisation et toxicité.** Le fait que 25,4 % des posts d'altérisation aient une toxicité modérée illustre un problème bien documenté des détecteurs de toxicité : ils repèrent les insultes et menaces explicites, mais manquent l'exclusion rhétorique sophistiquée. Un pipeline hybride combinant les deux signaux est plus robuste que chacun pris isolément.

**Limites du classificateur supervisé.** L'accord entre le détecteur rule-based (5,3 %) et le classificateur ML (5,29 %) est un signe ambigu. D'un côté, le modèle a bien appris les patterns. De l'autre, sa performance est bornée par le rappel du dictionnaire. Si le dictionnaire manque un type d'altérisation, le modèle le manquera aussi. Entraîner sur des annotations humaines indépendantes donnerait une mesure plus honnête des capacités réelles du modèle.

**Biais de composition du corpus.** Le corpus Berkeley a été conçu pour étudier les contenus nocifs, pas pour être représentatif des réseaux sociaux. Toutes les statistiques agrégées de ce rapport (taux de toxicité, taux d'altérisation) sont dominées par cette source. La comparaison directe avec les subreddits Reddit est méthodologiquement approximative. Une étude future gagnerait à séparer ces deux analyses plutôt qu'à les fusionner dans un seul corpus.

**La perte des timestamps Reddit.** Le champ `created_utc` était présent dans les données brutes mais n'a pas survécu au merge initial avec le corpus Berkeley. Cela a rendu impossible l'analyse de séries temporelles prévue en semaine 6. Une reconstruction partielle a été effectuée via Arctic Shift (un scraper alternatif des archives Reddit) pour patcher les dates, mais le volume récupéré était insuffisant pour des analyses temporelles significatives. Cette leçon plaide pour des tests d'intégrité colonne systématiques à chaque étape du pipeline.

---

## 6. Conclusion

Ce stage m'a permis de concevoir et de déployer un pipeline NLP complet de détection de l'altérisation sur 134 459 posts de réseaux sociaux. Le résultat principal est que la structure pronominale est un signal fort et économique : un taux d'altérisation de 22,2 % pour les posts combinant marqueurs WE et THEM, contre 1,2 % sans pronom. Le classificateur LinearSVC TF-IDF atteint un F1 de 0,990, et la modélisation BERTopic révèle que l'altérisation se concentre dans des thèmes identifiables (race, religion, géopolitique).

**Réponse à la problématique :** il est possible de construire un pipeline modulaire et reproductible qui détecte automatiquement l'altérisation avec une très bonne précision sur le contenu explicite. Les limites actuelles (altérisation implicite, langage codé, dépendance aux labels rule-based) sont aussi les pistes les plus concrètes pour une deuxième version.

**Compétences BUT mobilisées :**

| Compétence | Mise en œuvre concrète | Niveau avant | Niveau après |
|---|---|---|---|
| C1. Réaliser un développement | Pipeline Python de 6 modules, API HuggingFace, architecture incrémentale avec gestion d'erreurs | Sais coder des scripts Python isolés | Capable de concevoir une architecture multi-modules avec contrats d'interface clairs |
| C2. Optimiser des applications | Comparaison LinearSVC vs LR, traitement par lots (batch 32/64), cache UMAP, décision TF-IDF vs embeddings | Connais les bases des algorithmes ML | Sais évaluer le compromis précision/coût de calcul et justifier un choix de modèle |
| C3. Gérer des données | Corpus 134 459 lignes hétérogènes, pipeline CSV incrémental, BERTopic, dashboard Streamlit | Expérience sur petits datasets de TP | À l'aise sur des données volumineuses, réelles, bruitées et issues de sources multiples |
| C5. Conduire un projet | Planning 8 semaines en sprints, livrables hebdomadaires, rapport et présentation autonomes | Gestion de projet vue en TP courts | Premier projet long en autonomie : gestion des aléas, retards, arbitrages techniques |

Ce stage a confirmé mon intérêt pour le NLP et la data science appliquée aux sciences sociales. Il a aussi révélé des lacunes concrètes : la gestion de projets ML à long terme (versioning des données, reproductibilité) et les méthodes d'évaluation sur données déséquilibrées. Ce sont les points que je veux approfondir, notamment pendant mon échange à l'UQAC en 2026-2027.

Un bilan complet de mes forces et points de progression, ainsi que les facteurs environnementaux qui influenceront mon évolution professionnelle, est présenté en Annexe B (Matrice SWOT).

---

## 7. Glossaire

**Altérisation (othering)** : Procédé rhétorique consistant à construire un groupe externe (« eux ») comme menaçant, alien ou sous-humain, par opposition à un groupe interne légitime (« nous »). Théorisé par Tajfel & Turner (1979) dans la théorie de l'identité sociale.

**BERTopic** : Algorithme de modélisation thématique combinant des embeddings BERT, une réduction de dimension UMAP et un clustering HDBSCAN pour identifier des thèmes dans un corpus textuel.

**Corpus** : Ensemble structuré de textes servant de base à une analyse linguistique ou statistique.

**Embeddings** : Représentation vectorielle dense d'un texte dans un espace de grande dimension, capturant des relations sémantiques entre mots ou phrases.

**F1-score** : Moyenne harmonique de la précision et du rappel. Mesure équilibrée de la performance d'un classificateur binaire.

**GoEmotions** : Modèle BERT entraîné par Google sur des commentaires Reddit pour détecter 27 émotions fines.

**HDBSCAN** : Algorithme de clustering hiérarchique basé sur la densité, adapté aux données de grande dimension et aux clusters de tailles variables.

**Hate speech** : Discours incitant à la haine, à la discrimination ou à la violence envers un groupe de personnes.

**LinearSVC** : Machine à Vecteurs de Support à noyau linéaire, classificateur performant sur des données textuelles TF-IDF de haute dimension.

**NLP (Natural Language Processing)** : Ensemble de techniques informatiques permettant à une machine de traiter et comprendre le langage humain.

**Pipeline** : Chaîne de traitements séquentiels où la sortie d'une étape est l'entrée de la suivante.

**Pronoms WE/THEM** : Dans ce projet, désignent respectivement les marqueurs de groupe interne (we, us, our…) et de groupe externe (they, them, their, those people…).

**TF-IDF** : Term Frequency–Inverse Document Frequency. Représentation vectorielle d'un document basée sur la fréquence des mots pondérée par leur rareté dans le corpus.

**Toxicité** : Dans ce contexte, score produit par le modèle Detoxify estimant la probabilité qu'un texte soit perçu comme offensant ou nuisible.

**UMAP** : Algorithme de réduction de dimension non linéaire utilisé pour compresser les embeddings de haute dimension avant le clustering.

---

## 8. Bibliographie

Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The Pushshift Reddit dataset. *Proceedings of ICWSM 2020*.

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language. *Proceedings of ICWSM 2017*.

Del Tredici, M., & Fernández, R. (2019). You say hate speech, I say offensive language: An analysis of social factors and semantic relations. *Proceedings of RANLP 2019*.

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. *Proceedings of ACL 2020*.

Fortuna, P., & Nunes, S. (2018). A survey on automatic detection of hate speech in text. *ACM Computing Surveys, 51*(4), 1–30.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.

Hanu, L., & Unitary Team. (2020). Detoxify. Disponible sur : https://github.com/unitaryai/detoxify

Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application. *arXiv:2009.10277*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP 2019*.

Schmidt, A., & Wiegand, M. (2017). A survey on hate speech detection using natural language processing. *Proceedings of the Fifth International Workshop on NLP for Social Media*.

Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup conflict. In W. G. Austin & S. Worchel (Eds.), *The Social Psychology of Intergroup Relations* (pp. 33–47). Brooks/Cole.

van Dijk, T. A. (1993). *Elite Discourse and Racism*. Sage.

---

## 9. Annexes

### Annexe A : Utilisation de l'IA générative

Conformément aux exigences de l'IUT, je déclare ci-dessous les usages d'outils d'IA générative dans le cadre de ce projet et de ce rapport.

**Usage dans le projet (code) :**

| Outil | Usage | Part estimée |
|---|---|---|
| Claude (Anthropic) | Aide à la structuration du pipeline, débogage de scripts Python, suggestions d'architecture | ~15 % du code total |
| GitHub Copilot | Autocomplétion de code, génération de fonctions utilitaires | ~10 % du code total |

Tous les outputs générés ont été relus, testés et validés manuellement. Les logiques d'algorithme (choix des patterns, seuils de détection, évaluation des modèles) ont été entièrement déterminées par l'auteur.

**Bonnes pratiques d'utilisation de l'IA appliquées par l'entreprise / dans le projet :**

[À REMPLIR selon les pratiques réelles de l'organisme d'accueil. Structure suggérée :]
- L'IA était utilisée comme outil d'assistance et non de substitution : tout output généré était systématiquement relu, testé, et validé avant intégration.
- Les prompts étaient formulés de manière précise et contextualisée pour limiter les hallucinations.
- Aucune donnée sensible (données personnelles, données propriétaires) n'était transmise à des modèles cloud.
- Les résultats produits par IA étaient croisés avec d'autres sources (documentation officielle, résultats expérimentaux) avant d'être retenus.

**Usage dans ce rapport :**

| Outil | Usage | Part estimée |
|---|---|---|
| Claude (Anthropic) | Relecture orthographique, suggestions de reformulation sur certains passages | < 10 % du rapport |

L'ensemble des analyses, interprétations et conclusions sont originaux. Aucune section entière n'a été générée automatiquement.

**Principe de vérification :** Conformément aux exigences de l'IUT, je suis responsable de l'intégralité du contenu soumis à évaluation. Chaque information a été vérifiée, chaque source citée contrôlée, et chaque passage rédigé avec assistance IA relu et validé personnellement.

**Prompts principaux utilisés :**
- *« Comment structurer un pipeline NLP modulaire en Python pour une analyse séquentielle de corpus ? »*
- *« Quelle est la différence entre LinearSVC et LogisticRegression pour la classification de texte TF-IDF ? »*
- *« Relis ce paragraphe et propose des reformulations pour le rendre plus clair »*

---

### Annexe B : Matrice SWOT

| | Positif | Négatif |
|---|---|---|
| **Interne** | **Forces :** Maîtrise du pipeline Python de bout en bout. Résultats solides (F1=0,990). Livrables complets (code, dashboard, rapport, présentation). | **Faiblesses :** Lacunes en versioning des données ML. Gestion des timestamps sous-estimée. Peu d'expérience sur données déséquilibrées. |
| **Externe** | **Opportunités :** NLP et détection de contenu en forte demande. Outils open-source matures (HuggingFace, BERTopic). Échange UQAC pour approfondir le domaine. | **Menaces :** Évolution rapide des modèles (risque d'obsolescence des choix techniques). Complexité croissante du langage codé et des euphémismes. |
