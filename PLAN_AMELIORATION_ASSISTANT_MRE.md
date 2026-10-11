# Plan d'amelioration de l'assistant MRE

## Objectif

Faire evoluer le chat actuel en un assistant qui accompagne un utilisateur dans la preparation d'un projet au Maroc : comprendre son besoin, repondre avec des sources fiables, garder le contexte de son projet et l'aider a avancer etape par etape.

Ce fichier est une feuille de route. Les ameliorations ci-dessous ne sont pas toutes implementees.

## Etat constate dans le code

- Le chat propose actuellement quatre secteurs : immobilier, agriculture, industrie et tourisme ([templates/chat.html](templates/chat.html)).
- La recherche documentaire filtre les resultats par secteur et le prompt demande de ne pas repondre au-dela des documents retrouves ([main.py](main.py), [rag_engine.py](rag_engine.py)).
- L'historique est garde dans la memoire de la page web, mais n'est pas transmis avec la question au serveur ([templates/chat.html](templates/chat.html)). Le modele ne connait donc pas automatiquement les messages precedents.
- Les documents sources sont ranges dans des dossiers par secteur sous [data/](data/). Leur quantite, leur actualite et leur provenance doivent etre examinees secteur par secteur.
- Le chat et la page des opportunites sont deux fonctions distinctes. Le chat ne transforme pas encore une opportunite en plan de realisation suivi.

## Priorites

### 1. Definir le premier perimetre

Avant de modifier le code :

- Choisir les secteurs a couvrir dans la premiere version. Commencer par les quatre secteurs deja visibles ou en ajouter d'autres apres avoir prepare leurs sources.
- Definir les utilisateurs cibles : MRE, entrepreneur au Maroc, investisseur, ou plusieurs profils.
- Lister les questions auxquelles l'assistant doit aider pour chaque secteur : marche, terrain/local, creation d'entreprise, autorisations, fiscalite, financement, fournisseurs, recrutement, vente et export.
- Decider les livrables utiles : resume du projet, liste des demarches, questions a clarifier, risques et prochaines actions.

**Termine quand :** une liste des secteurs et des besoins prioritaires est validee avant le developpement.

### 2. Construire une base documentaire fiable et a jour

Pour chaque secteur :

- Inventorier les fichiers presents dans `data/<secteur>/` et noter les sujets couverts et manquants.
- Rechercher en priorite des sources officielles marocaines : administrations, textes de loi, agences publiques et organismes de financement. Faire valider les sources juridiques ou financieres par une personne competente.
- Enregistrer pour chaque document son titre, organisme, URL d'origine, date de publication, date de derniere verification, secteur, region et periode de validite si applicable.
- Remplacer ou signaler les statistiques anciennes. Ne pas presenter une aide, un appel ou une regle expiree comme toujours disponible.
- Ajouter une procedure de mise a jour : responsable, periodicite de verification et traitement des documents expires.
- Afficher la source et sa date dans les reponses, avec un lien quand il est disponible.

**Termine quand :** chaque secteur de lancement a une liste de sources approuvees, datees et verifiees, et les reponses permettent de retrouver ces sources.

### 3. Ajouter la memoire du projet utilisateur

Le serveur doit pouvoir retrouver le contexte apres un nouvel echange ou une reconnexion. Ajouter un profil de projet lie au compte de l'utilisateur, avec seulement les informations necessaires :

- nom ou description courte du projet ;
- secteur, ville ou region ;
- objectif et stade actuel ;
- budget indicatif et devise, si l'utilisateur choisit de le renseigner ;
- echeance et contraintes importantes ;
- decisions prises, questions ouvertes et prochaines actions.

Prevoir des controles pour consulter, corriger et supprimer les informations du projet. Ne pas demander de donnees sensibles inutiles.

**Termine quand :** le contexte est encore disponible apres rechargement de la page, et chaque utilisateur ne peut consulter que ses propres projets.

### 4. Transmettre le contexte utile au modele

- Envoyer au modele les derniers messages pertinents et un resume du projet, pas seulement la question la plus recente.
- Limiter la taille de l'historique et mettre a jour le resume au fil des conversations.
- Garder le secteur comme filtre principal, mais permettre la recherche de sujets transversaux necessaires au projet, par exemple creation d'entreprise, fiscalite ou financement.
- Conserver les citations associees aux documents effectivement recuperes. Si les sources sont insuffisantes ou contradictoires, le dire et demander une precision ou une verification humaine.

**Termine quand :** une question comme « et pour le financement ? » est comprise grace au contexte precedent, avec les sources correspondantes.

### 5. Guider la realisation par etapes

Organiser l'accompagnement autour d'un parcours adaptable, plutot que de repondre uniquement sous forme de questions-reponses :

1. Comprendre l'idee, la region, le budget et l'experience du porteur.
2. Clarifier les informations manquantes et evaluer les premieres conditions de faisabilite.
3. Presenter les demarches, autorisations et obligations pertinentes, avec sources et dates.
4. Identifier les pistes de financement ou opportunites correspondant au profil, sans promettre leur obtention.
5. Construire un plan d'action avec etapes, documents a preparer et points a confirmer.
6. A chaque retour de l'utilisateur, mettre a jour l'avancement et proposer la prochaine action utile.

Chaque recommandation importante doit distinguer les faits sourcés, les hypotheses et les informations a confirmer. L'assistant ne doit pas remplacer un avocat, un comptable, une banque ou une administration.

**Termine quand :** l'utilisateur peut retrouver les etapes accomplies, celles qui restent et les sources qui justifient les conseils.

### 6. Harmoniser les secteurs entre l'interface et l'API

- Definir une seule liste de secteurs partagee entre le frontend, l'API et les metadonnees d'ingestion.
- Afficher un secteur dans l'interface seulement si sa base documentaire est suffisamment prete.
- Ajouter une validation des valeurs de secteur et un message explicite lorsqu'un secteur n'est pas encore pris en charge.
- Tester qu'une recherche ne melange pas accidentellement les documents de secteurs differents.

**Termine quand :** chaque secteur visible peut etre selectionne, recherche correctement ses sources et dispose de tests.

### 7. Tester la qualite, la confidentialite et la securite

- Ecrire pour chaque secteur des questions de test avec les sources attendues et les reponses essentielles.
- Tester les cas sans source, avec source ancienne, avec sources contradictoires et avec une question hors sujet.
- Verifier que les citations renvoient aux bons documents et pages.
- Tester la reprise d'une conversation apres rechargement et la separation des projets entre deux comptes.
- Ne jamais exposer une conversation ou un projet prive a un autre utilisateur ou visiteur.
- Afficher clairement les limites des reponses juridiques, fiscales et financieres et les cas qui necessitent un professionnel.

**Termine quand :** les tests de reponse, de source, de reprise et d'isolation des comptes passent avant la mise en production.

## Ordre de realisation conseille

1. Valider les secteurs, les utilisateurs cibles et les questions prioritaires.
2. Auditer et completer les sources du premier secteur a lancer.
3. Ajouter le stockage securise du projet et de la conversation.
4. Transmettre le contexte au modele et ameliorer la recherche documentaire.
5. Construire le parcours et le plan d'action dans l'interface.
6. Ajouter les autres secteurs un par un, avec leurs sources et tests.
7. Faire tester le parcours complet par quelques utilisateurs avant ouverture plus large.

## Ce que tu peux preparer

- Choisir les secteurs de la premiere version.
- Donner, pour chaque secteur, des exemples de projets reels que tes utilisateurs veulent lancer.
- Rassembler les liens et documents officiels que tu connais, avec leur date si possible.
- Indiquer les regions marocaines prioritaires et les langues souhaitees (francais, arabe, autres).
- Decider quelles informations de projet l'utilisateur accepte d'enregistrer et pendant combien de temps.
- Demander une validation professionnelle pour les contenus juridiques, fiscaux ou financiers sensibles.

## Parcours de test utilisateur

1. Creer un compte et demarrer un projet dans un secteur disponible.
2. Decrire une idee avec une region et un budget indicatif.
3. Demander les demarches, puis poser une question de suivi sans repeter les informations deja donnees.
4. Recharger la page et verifier que le projet et son avancement sont conserves.
5. Verifier les sources, dates, prochaines actions et informations encore a confirmer.
6. Se connecter avec un autre compte et verifier que le premier projet n'est pas visible.

## Hors perimetre de cette feuille de route

Ce document ne modifie pas le comportement de l'application et ne garantit pas que les sources actuelles sont exhaustives ou juridiquement valides. Les sources et les reponses doivent etre verifiees avant tout usage professionnel ou toute decision d'investissement.