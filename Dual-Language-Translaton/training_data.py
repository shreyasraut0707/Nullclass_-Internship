"""
Training Data - Curated English-French and English-Hindi translation pairs
This dataset is used to fine-tune our translation models
"""

# English to French training pairs
# Each tuple: (English, French)
ENGLISH_FRENCH_PAIRS = [
    # Greetings and Common Phrases
    ("Good morning, how are you today?", "Bonjour, comment allez-vous aujourd'hui?"),
    ("Hello, my name is John and I am a student.", "Bonjour, je m'appelle John et je suis étudiant."),
    ("Good evening, welcome to our restaurant.", "Bonsoir, bienvenue dans notre restaurant."),
    ("Thank you very much for your help.", "Merci beaucoup pour votre aide."),
    ("Please have a seat and make yourself comfortable.", "Veuillez vous asseoir et vous mettre à l'aise."),
    ("Excuse me, could you please help me?", "Excusez-moi, pourriez-vous m'aider s'il vous plaît?"),
    ("I am very happy to meet you.", "Je suis très heureux de vous rencontrer."),
    ("Have a wonderful day ahead.", "Passez une merveilleuse journée."),
    ("See you tomorrow at the same time.", "À demain à la même heure."),
    ("It was a pleasure talking to you.", "C'était un plaisir de parler avec vous."),
    
    # Weather and Nature
    ("The weather is beautiful and sunny today.", "Le temps est beau et ensoleillé aujourd'hui."),
    ("It is raining heavily outside.", "Il pleut beaucoup dehors."),
    ("The mountains are covered with snow.", "Les montagnes sont couvertes de neige."),
    ("Spring is my favorite season of the year.", "Le printemps est ma saison préférée de l'année."),
    ("The flowers in the garden are blooming.", "Les fleurs du jardin sont en fleurs."),
    ("The sunset looks absolutely magnificent.", "Le coucher de soleil est absolument magnifique."),
    ("We should protect our natural environment.", "Nous devons protéger notre environnement naturel."),
    ("The river flows through the beautiful valley.", "La rivière coule à travers la belle vallée."),
    ("The forest is home to many wild animals.", "La forêt abrite de nombreux animaux sauvages."),
    ("Climate change is affecting our planet.", "Le changement climatique affecte notre planète."),
    
    # Education and Learning
    ("Education is the most powerful weapon to change the world.", "L'éducation est l'arme la plus puissante pour changer le monde."),
    ("I am studying computer science at university.", "J'étudie l'informatique à l'université."),
    ("Learning new languages opens many doors.", "Apprendre de nouvelles langues ouvre de nombreuses portes."),
    ("The teacher explained the lesson very clearly.", "Le professeur a expliqué la leçon très clairement."),
    ("Reading books helps improve knowledge and vocabulary.", "Lire des livres aide à améliorer les connaissances et le vocabulaire."),
    ("Students should complete their homework on time.", "Les étudiants doivent terminer leurs devoirs à temps."),
    ("The library has thousands of interesting books.", "La bibliothèque contient des milliers de livres intéressants."),
    ("Mathematics is an important subject for engineering.", "Les mathématiques sont une matière importante pour l'ingénierie."),
    ("She received excellent grades in all subjects.", "Elle a obtenu d'excellentes notes dans toutes les matières."),
    ("Knowledge is power and education is key.", "Le savoir est le pouvoir et l'éducation est la clé."),
    
    # Technology
    ("Technology is changing the world rapidly.", "La technologie change le monde rapidement."),
    ("Artificial intelligence is the future of computing.", "L'intelligence artificielle est l'avenir de l'informatique."),
    ("The internet connects people around the world.", "Internet connecte les gens du monde entier."),
    ("Mobile phones have become essential in our lives.", "Les téléphones portables sont devenus essentiels dans nos vies."),
    ("Software development requires creativity and logic.", "Le développement logiciel nécessite créativité et logique."),
    ("Machine learning models can recognize patterns.", "Les modèles d'apprentissage automatique peuvent reconnaître des modèles."),
    ("Data science is a growing field with many opportunities.", "La science des données est un domaine en croissance avec de nombreuses opportunités."),
    ("Cybersecurity protects our digital information.", "La cybersécurité protège nos informations numériques."),
    ("Cloud computing enables remote data storage.", "Le cloud computing permet le stockage de données à distance."),
    ("Programming languages help us communicate with computers.", "Les langages de programmation nous aident à communiquer avec les ordinateurs."),
    
    # Daily Life
    ("I usually wake up early in the morning.", "Je me réveille généralement tôt le matin."),
    ("Breakfast is the most important meal of the day.", "Le petit-déjeuner est le repas le plus important de la journée."),
    ("She goes to work by train every day.", "Elle va au travail en train tous les jours."),
    ("The shopping mall is crowded on weekends.", "Le centre commercial est bondé le week-end."),
    ("We should exercise regularly for good health.", "Nous devrions faire de l'exercice régulièrement pour une bonne santé."),
    ("Cooking at home is healthier than eating out.", "Cuisiner à la maison est plus sain que manger dehors."),
    ("The neighbors are very friendly and helpful.", "Les voisins sont très sympathiques et serviables."),
    ("Public transportation is convenient in this city.", "Les transports en commun sont pratiques dans cette ville."),
    ("Time management is crucial for productivity.", "La gestion du temps est cruciale pour la productivité."),
    ("A balanced lifestyle leads to happiness.", "Un mode de vie équilibré mène au bonheur."),
    
    # Travel and Places
    ("Paris is known as the city of lights.", "Paris est connue comme la ville des lumières."),
    ("The museum contains many historical artifacts.", "Le musée contient de nombreux artefacts historiques."),
    ("We are planning a vacation to the beach.", "Nous planifions des vacances à la plage."),
    ("The hotel offers excellent customer service.", "L'hôtel offre un excellent service client."),
    ("The airport is located outside the city.", "L'aéroport est situé en dehors de la ville."),
    ("Travel broadens the mind and creates memories.", "Voyager élargit l'esprit et crée des souvenirs."),
    ("The ancient castle attracts many tourists.", "Le château ancien attire de nombreux touristes."),
    ("We need to book train tickets in advance.", "Nous devons réserver les billets de train à l'avance."),
    ("The view from the mountain top is breathtaking.", "La vue depuis le sommet de la montagne est à couper le souffle."),
    ("Local cuisine is always worth trying.", "La cuisine locale vaut toujours la peine d'être essayée."),
    
    # Business and Work
    ("The company is expanding its operations globally.", "L'entreprise étend ses opérations à l'échelle mondiale."),
    ("Team collaboration leads to better results.", "La collaboration en équipe conduit à de meilleurs résultats."),
    ("The meeting will be held tomorrow afternoon.", "La réunion aura lieu demain après-midi."),
    ("Customer satisfaction is our top priority.", "La satisfaction client est notre priorité absolue."),
    ("The project deadline has been extended by a week.", "La date limite du projet a été prolongée d'une semaine."),
    ("Professional development is important for career growth.", "Le développement professionnel est important pour l'évolution de carrière."),
    ("The quarterly report shows positive growth.", "Le rapport trimestriel montre une croissance positive."),
    ("Effective communication is essential in business.", "Une communication efficace est essentielle en affaires."),
    ("Innovation drives competitive advantage.", "L'innovation stimule l'avantage concurrentiel."),
    ("Leadership requires vision and dedication.", "Le leadership nécessite vision et dévouement."),
    
    # Health and Wellness
    ("Regular exercise improves mental and physical health.", "L'exercice régulier améliore la santé mentale et physique."),
    ("Drinking plenty of water is essential for hydration.", "Boire beaucoup d'eau est essentiel pour l'hydratation."),
    ("A good night's sleep helps with recovery.", "Une bonne nuit de sommeil aide à la récupération."),
    ("Stress management techniques can reduce anxiety.", "Les techniques de gestion du stress peuvent réduire l'anxiété."),
    ("Healthy eating habits prevent many diseases.", "Des habitudes alimentaires saines préviennent de nombreuses maladies."),
    ("Mental health is as important as physical health.", "La santé mentale est aussi importante que la santé physique."),
    ("Meditation helps in achieving inner peace.", "La méditation aide à atteindre la paix intérieure."),
    ("Doctors recommend regular health checkups.", "Les médecins recommandent des bilans de santé réguliers."),
    ("Fresh fruits and vegetables are nutritious.", "Les fruits et légumes frais sont nutritifs."),
    ("Exercise releases endorphins that make us happy.", "L'exercice libère des endorphines qui nous rendent heureux."),
    
    # Family and Relationships
    ("Family time is precious and should be cherished.", "Le temps en famille est précieux et doit être chéri."),
    ("Children learn important values from their parents.", "Les enfants apprennent des valeurs importantes de leurs parents."),
    ("Strong relationships are built on trust and respect.", "Les relations solides sont fondées sur la confiance et le respect."),
    ("Grandparents have a wealth of wisdom to share.", "Les grands-parents ont une richesse de sagesse à partager."),
    ("Spending quality time together strengthens bonds.", "Passer du temps de qualité ensemble renforce les liens."),
    ("Communication is the foundation of healthy relationships.", "La communication est le fondement de relations saines."),
    ("Parents want the best future for their children.", "Les parents veulent le meilleur avenir pour leurs enfants."),
    ("Siblings often share a special lifelong bond.", "Les frères et sœurs partagent souvent un lien spécial pour la vie."),
    ("Love and kindness make a house a home.", "L'amour et la gentillesse font d'une maison un foyer."),
    ("Family traditions create lasting memories.", "Les traditions familiales créent des souvenirs durables."),
    
    # Culture and Arts
    ("Music has the power to bring people together.", "La musique a le pouvoir de rassembler les gens."),
    ("Art expresses emotions that words cannot describe.", "L'art exprime des émotions que les mots ne peuvent décrire."),
    ("The theater performance was absolutely stunning.", "La représentation théâtrale était absolument époustouflante."),
    ("Cultural diversity enriches our society.", "La diversité culturelle enrichit notre société."),
    ("Classical literature provides timeless wisdom.", "La littérature classique fournit une sagesse intemporelle."),
    ("Photography captures moments that last forever.", "La photographie capture des moments qui durent pour toujours."),
    ("Dance is a beautiful form of self expression.", "La danse est une belle forme d'expression de soi."),
    ("Film has the ability to inspire and educate.", "Le cinéma a la capacité d'inspirer et d'éduquer."),
    ("Preserving heritage is important for future generations.", "Préserver le patrimoine est important pour les générations futures."),
    ("Creativity knows no boundaries or limitations.", "La créativité ne connaît pas de frontières ni de limites."),
]

# English to Hindi training pairs
# Each tuple: (English, Hindi)
ENGLISH_HINDI_PAIRS = [
    # Greetings and Common Phrases
    ("Good morning, how are you today?", "सुप्रभात, आज आप कैसे हैं?"),
    ("Hello, my name is John and I am a student.", "नमस्ते, मेरा नाम जॉन है और मैं एक छात्र हूं।"),
    ("Good evening, welcome to our restaurant.", "शुभ संध्या, हमारे रेस्तरां में आपका स्वागत है।"),
    ("Thank you very much for your help.", "आपकी मदद के लिए बहुत बहुत धन्यवाद।"),
    ("Please have a seat and make yourself comfortable.", "कृपया बैठिए और आराम से रहिए।"),
    ("Excuse me, could you please help me?", "क्षमा करें, क्या आप मेरी मदद कर सकते हैं?"),
    ("I am very happy to meet you.", "मैं आपसे मिलकर बहुत खुश हूं।"),
    ("Have a wonderful day ahead.", "आगे के लिए शुभकामनाएं।"),
    ("See you tomorrow at the same time.", "कल उसी समय मिलते हैं।"),
    ("It was a pleasure talking to you.", "आपसे बात करके खुशी हुई।"),
    
    # Weather and Nature
    ("The weather is beautiful and sunny today.", "आज मौसम सुंदर और धूप वाला है।"),
    ("It is raining heavily outside.", "बाहर बहुत तेज बारिश हो रही है।"),
    ("The mountains are covered with snow.", "पहाड़ बर्फ से ढके हुए हैं।"),
    ("Spring is my favorite season of the year.", "वसंत मेरा पसंदीदा मौसम है।"),
    ("The flowers in the garden are blooming.", "बगीचे में फूल खिल रहे हैं।"),
    ("The sunset looks absolutely magnificent.", "सूर्यास्त बिल्कुल शानदार दिखता है।"),
    ("We should protect our natural environment.", "हमें अपने प्राकृतिक पर्यावरण की रक्षा करनी चाहिए।"),
    ("The river flows through the beautiful valley.", "नदी सुंदर घाटी से होकर बहती है।"),
    ("The forest is home to many wild animals.", "जंगल कई जंगली जानवरों का घर है।"),
    ("Climate change is affecting our planet.", "जलवायु परिवर्तन हमारे ग्रह को प्रभावित कर रहा है।"),
    
    # Education and Learning
    ("Education is the most powerful weapon to change the world.", "शिक्षा दुनिया को बदलने का सबसे शक्तिशाली हथियार है।"),
    ("I am studying computer science at university.", "मैं विश्वविद्यालय में कंप्यूटर विज्ञान पढ़ रहा हूं।"),
    ("Learning new languages opens many doors.", "नई भाषाएं सीखने से कई दरवाजे खुलते हैं।"),
    ("The teacher explained the lesson very clearly.", "शिक्षक ने पाठ बहुत स्पष्ट रूप से समझाया।"),
    ("Reading books helps improve knowledge and vocabulary.", "किताबें पढ़ने से ज्ञान और शब्दावली में सुधार होता है।"),
    ("Students should complete their homework on time.", "छात्रों को समय पर अपना होमवर्क पूरा करना चाहिए।"),
    ("The library has thousands of interesting books.", "पुस्तकालय में हजारों रोचक किताबें हैं।"),
    ("Mathematics is an important subject for engineering.", "गणित इंजीनियरिंग के लिए एक महत्वपूर्ण विषय है।"),
    ("She received excellent grades in all subjects.", "उसने सभी विषयों में उत्कृष्ट अंक प्राप्त किए।"),
    ("Knowledge is power and education is key.", "ज्ञान शक्ति है और शिक्षा कुंजी है।"),
    
    # Technology
    ("Technology is changing the world rapidly.", "तकनीक तेजी से दुनिया बदल रही है।"),
    ("Artificial intelligence is the future of computing.", "कृत्रिम बुद्धिमत्ता कंप्यूटिंग का भविष्य है।"),
    ("The internet connects people around the world.", "इंटरनेट दुनिया भर के लोगों को जोड़ता है।"),
    ("Mobile phones have become essential in our lives.", "मोबाइल फोन हमारे जीवन में आवश्यक हो गए हैं।"),
    ("Software development requires creativity and logic.", "सॉफ्टवेयर विकास के लिए रचनात्मकता और तर्क की आवश्यकता होती है।"),
    ("Machine learning models can recognize patterns.", "मशीन लर्निंग मॉडल पैटर्न को पहचान सकते हैं।"),
    ("Data science is a growing field with many opportunities.", "डेटा साइंस कई अवसरों वाला एक बढ़ता हुआ क्षेत्र है।"),
    ("Cybersecurity protects our digital information.", "साइबर सुरक्षा हमारी डिजिटल जानकारी की रक्षा करती है।"),
    ("Cloud computing enables remote data storage.", "क्लाउड कंप्यूटिंग दूरस्थ डेटा भंडारण को सक्षम बनाती है।"),
    ("Programming languages help us communicate with computers.", "प्रोग्रामिंग भाषाएं हमें कंप्यूटर के साथ संवाद करने में मदद करती हैं।"),
    
    # Daily Life
    ("I usually wake up early in the morning.", "मैं आमतौर पर सुबह जल्दी उठता हूं।"),
    ("Breakfast is the most important meal of the day.", "नाश्ता दिन का सबसे महत्वपूर्ण भोजन है।"),
    ("She goes to work by train every day.", "वह हर दिन ट्रेन से काम पर जाती है।"),
    ("The shopping mall is crowded on weekends.", "सप्ताहांत में शॉपिंग मॉल में भीड़ होती है।"),
    ("We should exercise regularly for good health.", "अच्छे स्वास्थ्य के लिए हमें नियमित व्यायाम करना चाहिए।"),
    ("Cooking at home is healthier than eating out.", "घर पर खाना बनाना बाहर खाने से ज्यादा स्वस्थ है।"),
    ("The neighbors are very friendly and helpful.", "पड़ोसी बहुत मिलनसार और मददगार हैं।"),
    ("Public transportation is convenient in this city.", "इस शहर में सार्वजनिक परिवहन सुविधाजनक है।"),
    ("Time management is crucial for productivity.", "उत्पादकता के लिए समय प्रबंधन महत्वपूर्ण है।"),
    ("A balanced lifestyle leads to happiness.", "संतुलित जीवनशैली खुशी की ओर ले जाती है।"),
    
    # Travel and Places
    ("India is known for its rich cultural heritage.", "भारत अपनी समृद्ध सांस्कृतिक विरासत के लिए जाना जाता है।"),
    ("The museum contains many historical artifacts.", "संग्रहालय में कई ऐतिहासिक कलाकृतियां हैं।"),
    ("We are planning a vacation to the beach.", "हम समुद्र तट पर छुट्टी की योजना बना रहे हैं।"),
    ("The hotel offers excellent customer service.", "होटल उत्कृष्ट ग्राहक सेवा प्रदान करता है।"),
    ("The airport is located outside the city.", "हवाई अड्डा शहर के बाहर स्थित है।"),
    ("Travel broadens the mind and creates memories.", "यात्रा मन को विस्तृत करती है और यादें बनाती है।"),
    ("The ancient temple attracts many tourists.", "प्राचीन मंदिर कई पर्यटकों को आकर्षित करता है।"),
    ("We need to book train tickets in advance.", "हमें पहले से ट्रेन टिकट बुक करने होंगे।"),
    ("The view from the mountain top is breathtaking.", "पहाड़ की चोटी से दृश्य लुभावना है।"),
    ("Local cuisine is always worth trying.", "स्थानीय व्यंजन हमेशा आजमाने लायक होते हैं।"),
    
    # Business and Work
    ("The company is expanding its operations globally.", "कंपनी वैश्विक स्तर पर अपना संचालन बढ़ा रही है।"),
    ("Team collaboration leads to better results.", "टीम सहयोग से बेहतर परिणाम मिलते हैं।"),
    ("The meeting will be held tomorrow afternoon.", "बैठक कल दोपहर को होगी।"),
    ("Customer satisfaction is our top priority.", "ग्राहक संतुष्टि हमारी सर्वोच्च प्राथमिकता है।"),
    ("The project deadline has been extended by a week.", "परियोजना की समय सीमा एक सप्ताह बढ़ा दी गई है।"),
    ("Professional development is important for career growth.", "व्यावसायिक विकास करियर वृद्धि के लिए महत्वपूर्ण है।"),
    ("The quarterly report shows positive growth.", "त्रैमासिक रिपोर्ट सकारात्मक वृद्धि दर्शाती है।"),
    ("Effective communication is essential in business.", "व्यापार में प्रभावी संचार आवश्यक है।"),
    ("Innovation drives competitive advantage.", "नवाचार प्रतिस्पर्धात्मक लाभ को बढ़ावा देता है।"),
    ("Leadership requires vision and dedication.", "नेतृत्व के लिए दृष्टि और समर्पण की आवश्यकता होती है।"),
    
    # Health and Wellness
    ("Regular exercise improves mental and physical health.", "नियमित व्यायाम मानसिक और शारीरिक स्वास्थ्य में सुधार करता है।"),
    ("Drinking plenty of water is essential for hydration.", "पर्याप्त पानी पीना हाइड्रेशन के लिए आवश्यक है।"),
    ("A good night's sleep helps with recovery.", "रात की अच्छी नींद रिकवरी में मदद करती है।"),
    ("Stress management techniques can reduce anxiety.", "तनाव प्रबंधन तकनीकें चिंता को कम कर सकती हैं।"),
    ("Healthy eating habits prevent many diseases.", "स्वस्थ खान-पान की आदतें कई बीमारियों को रोकती हैं।"),
    ("Mental health is as important as physical health.", "मानसिक स्वास्थ्य शारीरिक स्वास्थ्य जितना ही महत्वपूर्ण है।"),
    ("Meditation helps in achieving inner peace.", "ध्यान आंतरिक शांति प्राप्त करने में मदद करता है।"),
    ("Doctors recommend regular health checkups.", "डॉक्टर नियमित स्वास्थ्य जांच की सलाह देते हैं।"),
    ("Fresh fruits and vegetables are nutritious.", "ताजे फल और सब्जियां पौष्टिक होती हैं।"),
    ("Exercise releases endorphins that make us happy.", "व्यायाम एंडोर्फिन छोड़ता है जो हमें खुश करता है।"),
    
    # Family and Relationships
    ("Family time is precious and should be cherished.", "परिवार का समय अनमोल है और इसे संजोना चाहिए।"),
    ("Children learn important values from their parents.", "बच्चे अपने माता-पिता से महत्वपूर्ण मूल्य सीखते हैं।"),
    ("Strong relationships are built on trust and respect.", "मजबूत संबंध विश्वास और सम्मान पर बनते हैं।"),
    ("Grandparents have a wealth of wisdom to share.", "दादा-दादी के पास साझा करने के लिए ज्ञान का भंडार है।"),
    ("Spending quality time together strengthens bonds.", "साथ में गुणवत्तापूर्ण समय बिताने से बंधन मजबूत होते हैं।"),
    ("Communication is the foundation of healthy relationships.", "संचार स्वस्थ संबंधों की नींव है।"),
    ("Parents want the best future for their children.", "माता-पिता अपने बच्चों के लिए सबसे अच्छा भविष्य चाहते हैं।"),
    ("Siblings often share a special lifelong bond.", "भाई-बहन अक्सर जीवन भर का एक विशेष बंधन साझा करते हैं।"),
    ("Love and kindness make a house a home.", "प्यार और दया एक घर को घर बनाते हैं।"),
    ("Family traditions create lasting memories.", "पारिवारिक परंपराएं स्थायी यादें बनाती हैं।"),
    
    # Culture and Arts
    ("Music has the power to bring people together.", "संगीत में लोगों को एक साथ लाने की शक्ति है।"),
    ("Art expresses emotions that words cannot describe.", "कला उन भावनाओं को व्यक्त करती है जिन्हें शब्द बयां नहीं कर सकते।"),
    ("The theater performance was absolutely stunning.", "नाट्य प्रदर्शन बिल्कुल शानदार था।"),
    ("Cultural diversity enriches our society.", "सांस्कृतिक विविधता हमारे समाज को समृद्ध करती है।"),
    ("Classical literature provides timeless wisdom.", "शास्त्रीय साहित्य कालातीत ज्ञान प्रदान करता है।"),
    ("Photography captures moments that last forever.", "फोटोग्राफी उन पलों को कैद करती है जो हमेशा रहते हैं।"),
    ("Dance is a beautiful form of self expression.", "नृत्य आत्म अभिव्यक्ति का एक सुंदर रूप है।"),
    ("Film has the ability to inspire and educate.", "फिल्म में प्रेरित करने और शिक्षित करने की क्षमता है।"),
    ("Preserving heritage is important for future generations.", "विरासत को संरक्षित करना भावी पीढ़ियों के लिए महत्वपूर्ण है।"),
    ("Creativity knows no boundaries or limitations.", "रचनात्मकता कोई सीमा या बंधन नहीं जानती।"),
]


def get_training_data():
    """Return all training data pairs."""
    return {
        'en_fr': ENGLISH_FRENCH_PAIRS,
        'en_hi': ENGLISH_HINDI_PAIRS
    }


if __name__ == "__main__":
    # Print dataset statistics
    data = get_training_data()
    print(f"Training Data Statistics:")
    print(f"English-French pairs: {len(data['en_fr'])}")
    print(f"English-Hindi pairs: {len(data['en_hi'])}")
    print()
    print("Sample pairs:")
    print(f"EN: {data['en_fr'][0][0]}")
    print(f"FR: {data['en_fr'][0][1]}")
    print(f"HI: {data['en_hi'][0][1]}")
