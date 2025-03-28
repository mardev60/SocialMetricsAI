import sqlite3
import random
from src.database.database import insert_tweet

positive_tweets = [
    "J'ai adoré ce film, c'était vraiment excellent !",
    "Super expérience, je recommande vivement !",
    "Très satisfait de mon achat, qualité au rendez-vous.",
    "Service client exceptionnel, merci beaucoup !",
    "C'est génial, je suis très content du résultat.",
    "Produit de très bonne qualité, je recommande.",
    "Application facile à utiliser et très pratique.",
    "Excellente prestation, je reviendrai !",
    "Vraiment top, rien à redire.",
    "Agréablement surpris par la qualité du service.",
    
    "Ce restaurant est fantastique, la nourriture est délicieuse.",
    "Le meilleur café que j'ai jamais goûté !",
    "Plats savoureux et service impeccable.",
    "Cuisine raffinée et cadre agréable, parfait pour un dîner romantique.",
    "Les desserts sont à tomber par terre !",
    "Rapport qualité-prix imbattable pour ce restaurant.",
    "Ambiance chaleureuse et personnel attentionné.",
    "La pizza était parfaitement cuite, un délice !",
    "Menu varié et produits frais, que demander de plus ?",
    "Le brunch était copieux et délicieux.",

    "Ce film est un chef-d'œuvre, j'ai été captivé du début à la fin.",
    "Scénario brillant et acteurs talentueux.",
    "La meilleure série que j'ai vue cette année !",
    "Effets spéciaux impressionnants et histoire prenante.",
    "J'ai ri aux éclats tout au long du film.",
    "La bande sonore est magnifique et s'accorde parfaitement aux images.",
    "Un final de saison époustouflant, vivement la suite !",
    "Film touchant qui m'a fait verser quelques larmes.",
    "Réalisation soignée et photographie sublime.",
    "Les personnages sont attachants et bien développés.",

    "Ce smartphone est performant et l'appareil photo est excellent.",
    "Livraison rapide et produit bien emballé.",
    "L'interface est intuitive, même pour les débutants.",
    "Batterie qui tient toute la journée, c'est parfait !",
    "Le rapport qualité-prix est excellent pour ce produit.",
    "Service après-vente réactif et efficace.",
    "Design élégant et finitions de qualité.",
    "Fonctionne parfaitement, conforme à mes attentes.",
    "Installation facile et notice claire.",
    "Les mises à jour améliorent régulièrement l'expérience.",
    "Hôtel magnifique avec vue imprenable sur la mer.",
    "Personnel accueillant et aux petits soins.",
    "Chambre spacieuse, propre et bien équipée.",
    "Le petit-déjeuner était copieux et délicieux.",
    "Emplacement idéal, proche de toutes les attractions.",
    "Piscine agréable et bien entretenue.",
    "Séjour parfait, nous reviendrons avec plaisir.",
    "Rapport qualité-prix excellent pour cet hôtel.",
    "Cadre idyllique pour des vacances reposantes.",
    "Les excursions proposées par l'hôtel étaient fantastiques.",
    
    "C'est vraiment super !",
    "J'adore ce concept !",
    "Excellente initiative !",
    "Bravo pour cette réalisation !",
    "Merci pour ce moment agréable !",
    "Parfait, rien à redire !",
    "C'est exactement ce que je cherchais !",
    "Très belle surprise !",
    "Je suis comblé !",
    "Expérience à renouveler sans hésitation !",
    
    "Malgré quelques petits défauts, l'ensemble est très satisfaisant.",
    "Un peu cher mais la qualité est au rendez-vous.",
    "Service un peu lent mais la nourriture vaut l'attente.",
    "Certaines fonctionnalités manquent mais l'essentiel est là.",
    "Pas parfait mais largement au-dessus de la moyenne.",
    "Quelques longueurs mais l'histoire reste captivante.",
    "Un démarrage lent mais une fin extraordinaire.",
    "Légèrement bruyant mais très confortable.",
    "Interface perfectible mais fonctionnalités complètes.",
    "Un peu d'attente à l'accueil mais séjour parfait ensuite."
]

negative_tweets = [
    "Je suis très déçu par ce produit, ne perdez pas votre argent.",
    "Service client catastrophique, à éviter absolument.",
    "Expérience décevante, je ne reviendrai pas.",
    "Qualité médiocre pour un prix excessif.",
    "Horrible expérience, je déconseille fortement.",
    "Promesses non tenues, je me sens arnaqué.",
    "Prestation bien en-dessous de mes attentes.",
    "Rapport qualité-prix désastreux.",
    "Je regrette cet achat, ne faites pas la même erreur.",
    "Rien ne fonctionne comme prévu, c'est frustrant.",
    
    "Nourriture fade et service désastreux.",
    "Le plat était froid et mal présenté.",
    "Prix exorbitants pour des portions minuscules.",
    "Le serveur était impoli et peu attentif.",
    "La viande était trop cuite et sans saveur.",
    "Hygiène douteuse, je ne remettrai pas les pieds ici.",
    "Attente interminable pour être servi.",
    "Les ingrédients n'étaient pas frais du tout.",
    "Ambiance bruyante et tables trop serrées.",
    "Le menu est trompeur, rien à voir avec les photos.",
    
    "Film ennuyeux et prévisible, j'ai failli m'endormir.",
    "Scénario incohérent et plein de trous.",
    "Acteurs mauvais qui surjouent constamment.",
    "Effets spéciaux datés et peu convaincants.",
    "Histoire sans intérêt qui se traîne en longueur.",
    "Dialogues artificiels et mal écrits.",
    "La fin est bâclée et décevante.",
    "Personnages unidimensionnels et clichés.",
    "Réalisation amateur et montage chaotique.",
    "Une perte de temps, je ne recommande pas.",
    
    "Le produit est tombé en panne après deux semaines.",
    "Fragilité inquiétante, déjà plusieurs pièces cassées.",
    "Interface compliquée et peu intuitive.",
    "La batterie se décharge en quelques heures à peine.",
    "Impossible de joindre le service client.",
    "Livraison en retard et colis endommagé.",
    "Fonctionnalités limitées par rapport à la concurrence.",
    "Bugs fréquents qui rendent l'utilisation pénible.",
    "Design peu ergonomique et matériaux de mauvaise qualité.",
    "Mises à jour qui ralentissent l'appareil.",
    
    "Chambre sale et mal entretenue.",
    "Personnel désagréable et peu serviable.",
    "Bruit constant qui empêche de dormir.",
    "Petit-déjeuner minimal et de mauvaise qualité.",
    "Localisation éloignée de tout, impossible sans voiture.",
    "Climatisation défectueuse en pleine canicule.",
    "Photos trompeuses, la réalité est bien différente.",
    "Équipements vétustes et souvent hors service.",
    "Surréservation, nous avons dû changer d'hôtel.",
    "Mauvais rapport qualité-prix, bien trop cher pour ce que c'est.",
    
    "C'est vraiment nul !",
    "Je déteste ce concept !",
    "Quelle déception !",
    "À éviter à tout prix !",
    "Je ne recommande pas du tout !",
    "C'est une arnaque !",
    "Expérience catastrophique !",
    "Jamais plus !",
    "Quel gâchis !",
    "C'est inadmissible !",
    
    "Je n'ai pas du tout aimé ce film.",
    "Ce n'est pas du tout à la hauteur de mes attentes.",
    "Je ne recommanderais pas ce produit à mes amis.",
    "Ce restaurant n'est pas digne de sa réputation.",
    "L'hôtel n'était pas propre du tout.",
    "Le service n'a pas été à la hauteur du prix payé.",
    "Je n'ai pas apprécié l'attitude du personnel.",
    "Ce n'est pas un bon investissement.",
    "Je n'ai pas trouvé ce que je cherchais.",
    "Ce n'est pas acceptable pour un service premium.",
    
    "Je suis plutôt déçu de cette expérience.",
    "Le rapport qualité-prix laisse à désirer.",
    "Les performances sont en dessous de ce qu'on pourrait attendre.",
    "L'interface utilisateur manque cruellement d'intuitivité.",
    "Le service après-vente est difficile à joindre.",
    "La durée de vie de la batterie est insuffisante.",
    "Le confort des sièges laisse à désirer sur les longs trajets.",
    "La qualité de fabrication ne justifie pas ce prix.",
    "Les délais de livraison sont trop longs.",
    "Le site web est compliqué à naviguer.",
    "La connexion internet était instable pendant tout le séjour.",
    "Les mises à jour fréquentes sont pénibles.",
    "Le bruit environnant gâche l'expérience.",
    "Les options sont trop limitées.",
    "La documentation fournie est insuffisante.",
    "Le temps d'attente est excessif.",
    "Les frais cachés sont nombreux.",
    "La politique de retour est trop restrictive.",
    "Le support technique n'a pas résolu mon problème.",
    "L'application plante régulièrement.",
    
    "Je trouve que ce produit ne répond pas aux attentes.",
    "Mon expérience avec ce service a été globalement insatisfaisante.",
    "Je ne pense pas renouveler mon abonnement l'année prochaine.",
    "La qualité s'est dégradée au fil du temps.",
    "Les promesses marketing ne correspondent pas à la réalité.",
    "J'ai rencontré plusieurs problèmes lors de l'utilisation.",
    "Le rapport qualité-prix n'est pas justifié selon moi.",
    "Les fonctionnalités manquent de finition.",
    "L'expérience utilisateur pourrait être grandement améliorée.",
    "Je m'attendais à mieux compte tenu de la réputation.",
    "La fiabilité laisse à désirer sur le long terme.",
    "Les performances diminuent rapidement avec l'usage.",
    "Le design présente des défauts ergonomiques évidents.",
    "Le service client m'a laissé sans solution.",
    "Les matériaux utilisés semblent de qualité inférieure.",
    "L'autonomie est bien inférieure à celle annoncée.",
    "Les délais ne sont jamais respectés.",
    "La maintenance est compliquée et coûteuse.",
    "Les mises à jour n'apportent pas d'améliorations significatives.",
    "L'interface est confuse et mal organisée."
]

neutral_tweets = [
    "Le produit correspond à la description, ni plus ni moins.",
    "Service correct, dans la moyenne.",
    "Qualité standard pour ce type de produit.",
    "Fonctionnalités basiques mais suffisantes pour mon usage.",
    "Ni déçu ni impressionné, c'est convenable.",
    "Prix dans la moyenne du marché.",
    "Livraison dans les délais annoncés.",
    "Installation sans problème particulier.",
    "Interface classique, sans originalité.",
    "Conforme à mes attentes modérées.",
    
    "Certains aspects sont bons, d'autres décevants.",
    "Points forts : design et prix. Points faibles : durabilité et performance.",
    "Bon rapport qualité-prix mais service après-vente à améliorer.",
    "Bonne nourriture mais service lent et ambiance bruyante.",
    "Film avec d'excellentes scènes mais aussi des longueurs ennuyeuses.",
    "Hôtel bien situé mais chambres vieillissantes.",
    "Application utile mais souffre de quelques bugs.",
    "Personnel agréable mais infrastructures à moderniser.",
    "Bon concept mais mise en œuvre perfectible.",
    "Début prometteur mais fin décevante.",
    
    "Est-ce que quelqu'un a testé la version premium ?",
    "Je me demande si ça vaut le coup d'attendre la prochaine version.",
    "Quelqu'un sait si c'est compatible avec mon appareil ?",
    "J'hésite entre ce modèle et le concurrent, des avis ?",
    "Est-ce que ça fonctionne aussi bien qu'ils le prétendent ?",
    "Je n'ai pas encore décidé si je vais y retourner.",
    "Difficile de se faire une opinion après une seule utilisation.",
    "Je réserve mon jugement jusqu'à un test plus approfondi.",
    "Pas encore sûr si l'investissement en vaut la peine.",
    "Je dois l'utiliser plus longtemps pour me faire un avis définitif.",
    
    "Le restaurant propose une cuisine traditionnelle française.",
    "L'hôtel est situé à 10 minutes à pied de la plage.",
    "Le film dure 2h15 et est classé tous publics.",
    "L'application est disponible sur iOS et Android.",
    "Le produit est garanti 2 ans par le fabricant.",
    "Le menu change selon les saisons.",
    "La batterie a une capacité de 4000mAh.",
    "L'écran mesure 6,5 pouces de diagonale.",
    "Le service client est ouvert du lundi au vendredi de 9h à 18h.",
    "La livraison est gratuite à partir de 50€ d'achat."
]

negation_tweets = [
    "Je n'ai pas aimé ce film du tout.",
    "Ce n'est pas un bon restaurant, à éviter.",
    "Je n'ai pas été satisfait de mon séjour.",
    "Ce produit n'est pas à la hauteur de mes attentes.",
    "Je n'ai pas trouvé le personnel accueillant.",
    "Ce n'est pas une expérience que je souhaite renouveler.",
    "Je n'ai pas apprécié la qualité de la nourriture.",
    "Ce n'est pas un achat que je recommanderais.",
    "Je n'ai jamais vu un service aussi mauvais.",
    "Ce n'est pas du tout pratique à utiliser.",
    "Horrible, je n'ai pas aimé du tout.",
    "Catastrophique, je n'ai pas apprécié l'expérience.",
    "Décevant, je n'ai pas trouvé ça à la hauteur.",
    "Médiocre, je n'ai pas été convaincu.",
    "Terrible, je n'ai pas passé un bon moment.",

    "Ce n'est pas mauvais du tout, plutôt agréable même.",
    "Je n'ai pas été déçu par la qualité du service.",
    "Ce n'est pas désagréable comme expérience.",
    "Le film n'était pas aussi ennuyeux que je le craignais.",
    "Le repas n'était pas aussi cher que prévu.",
    "Ce n'est pas la pire application que j'ai utilisée.",
    "Je n'ai pas regretté mon achat finalement.",
    "Ce n'est pas aussi compliqué à utiliser que je le pensais.",
    "Le personnel n'était pas impoli, au contraire.",
    "Ce n'est pas un mauvais choix pour ce prix.",
    "Ce n'est pas terrible mais ça fait le job.",
    "Je n'ai pas détesté, c'était même plutôt bien.",
    "Ce n'est pas si mal que ça en fin de compte.",
    "Je n'ai pas eu de mauvaise surprise, tout fonctionnait bien.",
    "Ce n'est pas un échec, je suis même assez satisfait.",
    
    "Je ne peux pas dire que j'ai apprécié cette expérience.",
    "On ne peut pas qualifier ce service de satisfaisant.",
    "Je ne saurais recommander ce produit à qui que ce soit.",
    "Ce n'est vraiment pas ce à quoi je m'attendais, quelle déception.",
    "Je ne comprends pas comment ce restaurant peut avoir de bonnes critiques.",
    "Ce n'est en aucun cas un bon investissement.",
    "Je ne vois pas l'intérêt de ce produit.",
    "Ce n'est absolument pas à la hauteur du prix demandé.",
    "Je ne retournerai certainement pas dans cet établissement.",
    "Ce n'est pas du tout conforme à la description.",
    "Je ne pense pas qu'il soit judicieux d'acheter ce produit.",
    "Ce n'est pas ce que j'appellerais un service de qualité.",
    "Je ne trouve aucune qualité à ce produit.",
    "Ce n'est pas du tout ergonomique ni pratique.",
    "Je ne vois pas comment on peut apprécier cette interface.",
    
    "Ce n'est pas sans qualités, j'ai été agréablement surpris.",
    "Je ne dirais pas que c'est mauvais, au contraire.",
    "Ce n'est pas tous les jours qu'on trouve un service aussi efficace.",
    "On ne peut pas nier que le rapport qualité-prix est excellent.",
    "Je ne regrette absolument pas cet achat.",
    "Ce n'est pas souvent que je suis aussi satisfait d'un produit.",
    "Je ne peux pas me plaindre de la qualité du service.",
    "Ce n'est pas exagéré de dire que c'est l'un des meilleurs restaurants de la ville.",
    "Je n'ai pas trouvé de défaut majeur à signaler.",
    "Ce n'est pas pour rien que ce produit est si populaire.",
    "Je ne saurais trop recommander ce service.",
    "Ce n'est pas facile de trouver mieux pour ce prix.",
    "Je n'ai jamais eu une expérience aussi fluide.",
    "Ce n'est pas tous les jours qu'on voit une telle qualité.",
    "Je ne peux pas imaginer une meilleure solution pour mes besoins."
]

idiomatic_tweets = [
    "Ce film, c'est la goutte d'eau qui fait déborder le vase.",
    "Ce service client, c'est vraiment la croix et la bannière.",
    "Leur politique de retour, c'est un vrai parcours du combattant.",
    "Cette application tourne comme une patate chaude.",
    "Ce restaurant, c'est la soupe à la grimace.",
    "Avec ce produit, j'ai l'impression de me faire rouler dans la farine.",
    "Cette entreprise nous mène en bateau depuis des mois.",
    "Ce téléphone m'a coûté les yeux de la tête pour rien.",
    "Leur service après-vente, c'est comme parler à un mur.",
    "Cette mise à jour a mis le système sens dessus dessous.",
    "Leur nouvelle politique, c'est vraiment la goutte d'eau.",
    "Ce film est à dormir debout, quel ennui !",
    "Ce jeu vidéo, c'est du réchauffé.",
    "Cette série tombe à plat dès le deuxième épisode.",
    "Leur nouvelle interface, c'est le monde à l'envers.",
    
    "Ce restaurant, c'est la cerise sur le gâteau de notre séjour.",
    "Cette application, c'est la solution qui tombe à pic.",
    "Ce film, c'est un vrai coup de cœur.",
    "Leur service client, c'est aux petits oignons.",
    "Ce produit, c'est le jour et la nuit comparé au précédent.",
    "Cette expérience, c'est du pain béni pour les amateurs.",
    "Ce concert, c'était vraiment d'enfer !",
    "Cette fonctionnalité, c'est la cerise sur le gâteau.",
    "Ce restaurant, on s'y régale, c'est un vrai délice.",
    "Cette mise à jour, c'est un vent de fraîcheur.",
    "Ce nouveau modèle, c'est un bijou de technologie.",
    "Cette série, c'est un vrai régal pour les yeux.",
    "Ce jeu, c'est une vraie pépite.",
    "Cette interface, c'est un jeu d'enfant à utiliser.",
    "Ce service, c'est le top du top."
]

def generate_dataset():
    print("Génération du dataset de tweets...")
    
    count = 0
    
    for tweet in positive_tweets:
        try:
            insert_tweet(tweet, 1, 0)
            count += 1
        except Exception:
            pass
    
    for tweet in negative_tweets:
        try:
            insert_tweet(tweet, 0, 1)
            count += 1
        except Exception:
            pass
    
    for tweet in neutral_tweets:
        try:
            positive = random.choice([0, 0, 0, 1]) 
            negative = 1 - positive
            insert_tweet(tweet, positive, negative)
            count += 1
        except Exception:
            pass
    
    for i, tweet in enumerate(negation_tweets):
        try:
            if i < len(negation_tweets) // 2:
                insert_tweet(tweet, 0, 1)
            else:
                insert_tweet(tweet, 1, 0)
            count += 1
        except Exception:
            pass
    
    for i, tweet in enumerate(idiomatic_tweets):
        try:
            if i < len(idiomatic_tweets) // 2:     
                insert_tweet(tweet, 0, 1)
            else:
                insert_tweet(tweet, 1, 0)
            count += 1
        except Exception:
            pass
    
    print(f"Dataset généré avec succès ! {count} tweets insérés dans la base de données.")
    print("Vous pouvez maintenant réentraîner votre modèle avec 'python -m src.scripts.retrain'")

if __name__ == "__main__":
    generate_dataset() 