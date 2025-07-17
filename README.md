# ML_Final_Project

kaggle ის competition: Walmart Recruiting - Store Sales Forecasting

კონკურსის მიზანი გახლავთ ვალმარტის სხვადასხვა მაღაზიის ყოველკვირეული ჩანაწერების საფუძველზე ვივარაუდოთ Weekly_Sales სვეტის მნიშვნელობები. პრობლემა მიეკუთვნება Time-Series Problem-ს რასთან მიმართებითაც გავტესტეთ რამდენიმე სხვადასხვა ტიპის მოდელი და არქიტექტურა. 

პირველ რიგში განვახორციელეთ მონაცემების დეტალური ანალიზი, რის საფუძველზეც მივედით მნიშვნელოვან დასკვნებამდე. გრაფიკის აგებით აშკარად გამოიკვეთა, რომ Weekly_Sales საშუალოს ახასიათებს წლიური ციკლურობა. თუმცა DATE თან ერთად აუცილებელია მხედველობაში მივიღოთ სხვა სვეტების მნიშვნელობებიც. ამ სვეტთაგან ყველაზე მნიშვნელოვანი IsHoliday სვეტია, თუმცა მხოლოდ ამ სვეტით მოდელი ვერ დაიჭერს შესაბამის ტრენდს ამიტომ, feature engineering ნაწილში დავამატეთ ცალკე სვეტები, რომლებიც შეესაბამებიან კონკრეტული დღესასწაულის წინა, ახლანდელ და შემდეგ კვირას. 
45 Store ის ინდივიდუალური ანალიზის შედეგად გამოიკვეთა თითქმის იდენტური ტენდენცია, გარდა რამდენიმე მაღაზიისა (28, 30, 33, 36, 42, 43 და კიდევ რამდენიმე მნიშვნელოვნად განსხვავდება უმრავლესობისგან.) მიუხედავად გასხვავებისა, მაღაზიების უმეტესობაში ტენდენცია წლიდან წლამდე ნარჩუნდება, რის გამოც მიზანშეწონილად მივიჩნიეთ feature engineering ით ლაგების დამატება, რომელიც 1 წლის ან 2 წლის წინანდელ მონაცემს ინახავს.
ასევე დავამატეთ sin და cos fourier feature ები რომლებიც თავის მხრივ დაეხმარა მოდელს სეზონურობის უკეთ აღქმაში.

რეპოზიტორიის სტრუქტურა:
    experiments ფოლდერში მოცემულია სხვადასხვა მოდელის ექსპერიმენტები, რომლებშიც კარგად ჩანს სხვადასხვა მიდგომებისა და მოდელების გატესტვის მცდელობა.

model_experiment_XGBoost: 
    
    საწყისი მოდელია, რომელშიც ძალიან basic feature-ებია და ჰიპერპარამეტრების ოპტიმიზაციასაც დიდი მნიშვნელობა არ ჰქონდა. 
    პარამეტრები და მეტრიკები ასე გამოიყურება:
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42
        )
        ქროს ვალიდაციით მიღებული შედეგები:
            Mean Train RMSE: 5812.36
            Mean Train WMAE: 3326.78
            Mean Valid RMSE: 6330.36
            Mean Valid WMAE: 3659.47

model_experiment_xgboost2:

    ში დავამატე feature ები, როგორიცაა: Year, Month, Week, Day, DayOfWeek, PrevYearSales ამათი დამატებით მოდელის პერფორმანსი საგრძნობლად გაუმჯობესდა. 
        model = XGBRegressor(
            n_estimators=400,
            max_depth=7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42
        )
        ქროს ვალიდაციით მიღებული შედეგები:
            Mean Train RMSE: 2471.23
            Mean Train WMAE: 1491.55
            Mean Valid RMSE: 4771.06
            Mean Valid WMAE: 2448.34
        კაგელზე საბმიშენი: დაახლოებით 3745

model_experiment_xgboost3:

    ში დავამატე fourier feature ები, isWeekBefore{Holiday}, isWeekAfter{Holiday} და ჰიპერპარამეტრები დავატუნინგე, რის შედეგადაც მოდელის პერფორმანსი უფრო მეტად გაუმჯობესდა.
    საუკეთესო მოდელი დალოგილია dagshub-ზე: https://dagshub.com/Givi-Modebadze/Final_Project_ML.mlflow/#/experiments/0

        model = XGBRegressor(
            n_estimators=500,
            min_child_weight=10,
            learning_rate=0.1,
            max_depth=7,
            n_jobs=-1,
            random_state=42
        )
        ქროს ვალიდაციით მიღებული საუკეთესო მოდელის შედეგები:
            Mean Train RMSE: 3380.31
            Mean Train WMAE: 1932.04
            Mean Valid RMSE: 4443.44
            Mean Valid WMAE: 2167.41
        კაგელზე საბმიშენი: დაახლოებით 3280

model_experiment_TFT:

    ში მოცემულია Temporal Fusion Transformer არქიტექტურა, რომელიც გასამართად საკმაოდ რთული აღმოჩნდა, ამიტომ მხოლოდ basic არქიტექტურაა. ასევე საკმაოდ დიდი დრო სჭირდებოდა სრულ მონაცემებზე ტრენინგს ამიტომ პირველი რიგში პერფორმანსის შესამოწმებლად მხოლოდ პირველ მაღაზიაზე გავტესტე. ვალიდაციაზე შედეგები ასე გამოიყურებოდა:
        MAE: 20366.52
        RMSE: 28377.04
        WMAE: 20366.52
    ამის გამო გადავწყვიტე, რომ მოდელისთვის ჰიპერპარამეტრების დატუნინგება დიდ ვერაფერ შედეგს მოგვცემდა. იმის გათვალისწინებით, თუ რამხელა დრო სჭირდება თითოეულ გაშვებას.


random_forest:

    სანამ ამ კომპლექსურ მოდელებზე მუშაობას დავიწყებდი ვიფიქრე, რომ რაიმე basic model შემექმნა. მონაცემების და თვითონ შეჯიბრის უკეთესად შესაცნობად გადავწყვიტე random_forest მოდელი დამეტრეინინგებინა მიუხედავად იმისა, რომ მოთხოვნებში არ ეწერა. საინტერესოა, რომ მან არც ისე ცუდი შეეგი დადო.
    
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        max_depth=30,
        n_jobs=-1
    )
    
    აქ ვცადე მეპოვა კარგი ჰიპერპარამეტრები overfit-ის თავიდან ასაცილებლად ამ მოდელმა კარგად იმუშავა. უმეტესად 3ივე run-ის შედეგი მსგავსი იყო და ყველაზე კარგმა ეს დადო:
        Mean Train RMSE: 1450.62
        Mean Train WMAE: 621.21
        Mean Valid RMSE: 4112.44
        Mean Valid WMAE: 1905.06
    კეგლზე საბმიშენი: 6825

old_lightGBM:

    აქ მანამდე დავიწყე მუშაობა, სანამ გადავწყვიტავდით, თუ როგორ უნდა გაგვეყო მონაცემები და feature engineering როგორ უნდა გაგვეკეთებინა. აქედან გამომდინარე მარტივი შესამჩნევია, რომ საეჭვოდ კარგი შედეგები დადო თვითონ notebook-ში. ამ პარამეტრებით:
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    
    დადო ეს შედეგი:
        Mean Train RMSE: 3383.45
        Mean Train WMAE: 1776.05
        Mean Valid RMSE: 2946.82
        Mean Valid WMAE: 1530.10
    აქ იმითაც კი ჩანს, რომ კარგი შედეგი არ არის, რომ ვალიდაციაზე უკეთესი შედეგი დადო ვიდრე თრეინინგზე(kaggle-ზე დადო 19820 რაც აბსურდულია). ამას შემდეგ ფაილში გამოვასწორებ.

lightGBM:

    აქ უკვე დავამატე feature engineering და ვცადე underfit გამომესწორებინა learning rate-ის, num_leaves-ისა და feature_fraction-ის გაზრდით. ამან სასურველ შედეგს მიაღწია. შემდეგი პარამეტრებით:
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 150,
        'learning_rate': 0.15,
        'feature_fraction': 1,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2
    მივიღე ეს შედეგი:
        Mean Train RMSE: 1793.44
        Mean Train WMAE: 2113.45
        Mean Valid RMSE: 3723.63
        Mean Valid WMAE: 1971.56
    kaggle-ზე დაიდო ეს შედეგი:
        Score: 3476.58238
        Private score: 3544.26638

Arima/Sarima:

    აქ უკვე გვჭირდებოდა ჩვენი ვრაპერი კლასი შეგვექმნა, რომელიც ცალკე arima/sarima მოდელს შექმნიდა თითოეული store-სთვის(დეპარტამენტებისთვისაც ვცადე, მაგრამ ტრეინინგი იმდენ ხანს უნდებოდა იქრაშებოდა). საბოლოო ჯამში arima-მ არ იმუშავა სასურველად, იმიტომ რომ ეს სპეციალური დღეები, როგორიცაა ახალი წელი,მადლიერების დღე და ა.შ. ძალიან სწრაფად ხდება. ეს მოდელები ელოდებიან ნელ-ნელა ცვლილებას, მაგალითად ზაფხულისკებ უფრო მეტი ადამიანი ყიდულობს ბიკინის, ვიდრე ზამთარში. შედეგად ამ მოდელებმა ეს შედეგი დადეს:
        Mean Train RMSE: 22888.12
        Mean Train WMAE: 15393.41
        Mean Valid RMSE: 22117.13
        Mean Valid WMAE: 15230.82
    ეს რა თქმა უნდა არ არის სასურველი, მაგრამ ეს მოდელები უბრალოდ ვერ მოერგნენ ჩვენს მოთხოვნებს.

DLinear:

    ეს მოდელი საკმაოდ შეესაბამებოდა ჩვენს მოთხოვნებს, ვინაიდან kernel_size-ის ცვლილებით შეგვიძლია დავარეგულიროთ, თუ რამხელა დროის პერიოდში დააკვირდეს ცვლილებებს. მაგრამ ვინაიდან ეს მოდელი არ არის ძალიან sophisticated და ხშირად გამოიყენება baseline მოდელად ძალიან გასაოცარი შედეგები არ დადო.
    ამ ჰიპერპარამეტრებით:
        'seq_len': 52,
        'kernel_size': 10,
    მან დადო ეს შედეგი:
        Mean Train RMSE: 16544.83
        Mean Train WMAE: 9872.55
        Mean Valid RMSE: 6850.76
        Mean Valid WMAE: 5413.62

model_experiment_PatchTST: 

    ეს მოდელი გასამართად საკმაოდ რთული აღმოჩნდა, თუმცა შესაბამისი ბიბლიოთეკების დაყენებით საბოლოოდ არც ისე დიდი კოდი გამოვიდა. 
    ბასიქ მოდელმაც კი ძალიან კარგი შედეგი აჩვენა ყოველგვარი პრეპროცესინგის და feature engineering ის გარეშე უბრალოდ NeuralForecast გამოვიყენე და ვრაპერი დავუწერე. 
    კარგი შედეგის გამო მაქსიმალურად დავატუნინგე ჰიპერპარამეტრები, რითაც საბოლოოდ ასეთი შედეგი მივიღეთ:
    
        model = PatchTST(
            h=60,
            input_size=52,
            patch_len=16,
            batch_size=64,
            stride=4,
            n_heads=8,
            dropout=0.1,
            max_steps=5000,
            learning_rate=0.0005,
            activation='relu',
            enable_progress_bar=True
        )
    ვალიდაციაზე შედეგები:
        WMAE: 2581.45
        RMSE: 5966.96
    კაგლზე შედეგები:
        private score: 2777.38
        public score: 2695.34
    mlflow link:
        https://dagshub.com/Givi-Modebadze/Final_Project_ML.mlflow/#/experiments/1

model_experiment_NBEATS:

    ეს მოდელიც საკმაოდ რთული იყო გამოსაყენებლად, მაგრამ NeuralForecast-მა აქაც გვიშველა და იგივე ვრაპერი გამოვიყენე აქაც რაც წინა მოდელში. 
    ამასაც კარგი შედეგები ჰქონდა და ამიტომ დავატუნინგე ჰიპერპარამეტრები მაქსიმალურად. საბოლოოდ მივიღეთ ეს მოდელი:
    
        model = NBEATS(
            h=60,
            input_size=52,
            max_steps=4000,
            learning_rate=0.0005,
            stack_types=['trend', 'seasonality'],
            n_blocks = [2,2],
            mlp_units=[[256, 256], [256, 256]],
            activation='LeakyReLU',
        )
        
    ვალიდაციაზე შედეგები:
        WMAE: 2682.76
        RMSE: 6073.24
    კაგლზე შედეგები:
        private score: 2848.49
        public score: 2763.07
    mlflow link:
        https://dagshub.com/Givi-Modebadze/Final_Project_ML.mlflow/#/experiments/2
        
model_inference:

    ამ ფაილში ვაგენერირებთ ტესტ სეტზე საბმიშენს საუკეთესო მოდელის გამოყენებით. 
    საუკეთესო მოდელად შერჩა PatchTST. 
    
