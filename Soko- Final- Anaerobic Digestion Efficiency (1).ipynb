{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Hi, welcome to my (Luke Soko) final project for ABE 516X. There are ~2,000,000 farms across the US, but only ~330 operating manure-based anaerobic digesters. Why? They're historically unprofitable (until recently for specific scenarios). My research models anaerobic digestion profitability. My goal is to publish profitable scenarios for anaerobic digester use to stimulate anaerobic digester implementation and reduce agricultural greenhouse gas emissions. \n",
    "\n",
    "One component I must understand is anaerobic digester efficiency. The feedstocks of anaerobic digesters are typically characterized by biochemical methane potential (BMP) which is generally in units of (L CH4/g volatile solids). The BMP value can be considered the best case scenario for gas production, or 100% digestion efficiency. The Livestock Offest Protocol calculator from the California Air Resources Board features BMPs of manure. Using BMPs, the ASAE Manure Production Standard (D384.2), and assuming a 60% CH4 content in biogas, I calculated that a dairy cow produces a maximum of 101.3 ft3 biogas/day and a swine produces a maximum of 5.3 ft3 biogas/day.\n",
    "\n",
    "EPA AgSTAR has a database featuring number of dairy, swine, and daily biogas production (among other things) for operating anaerobic digesters across the US. I compare the daily biogas production from operating digesters to the hypothetical maximum biogas production to calculate an anaerobic digestion efficiency for different types of anaerobic digesters. I also use the AgSTAR database for machine learning to predict the daily biogas production for inputted number of dairy cows and digester type.\n",
    "\n",
    "Over 90% of anaerobic digesters are characterized as complete mix, plug flow, or impermeable covered lagoon anaerobic digesters. Therefore, these three types of digesters will be analyzed. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "agstar = pd.read_csv(\"agstar2.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Project Name', 'Project Type', 'City', 'County', 'State',\n",
       "       'Digester Type', 'Status', 'Year Operational', 'Animal/Farm Type(s)',\n",
       "       'Cattle', 'Dairy', 'Poultry', 'Swine', 'Co-Digestion',\n",
       "       ' Biogas Generation Estimate (cu_ft/day) ',\n",
       "       ' Electricity Generated (kWh/yr) ', 'Biogas End Use(s)',\n",
       "       'System Designer(s)_Developer(s) and Affiliates', 'Receiving Utility',\n",
       "       ' Total Emission Reductions (MTCO2e/yr) ', 'Awarded USDA Funding?'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "agstar.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import statsmodels.formula.api as smf\n",
    "%matplotlib inline\n",
    "import sklearn\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.svm import SVC, LinearSVC\n",
    "from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, mean_squared_error\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split \n",
    "from sklearn.tree import DecisionTreeClassifier \n",
    "import seaborn as sns\n",
    "import statsmodels.api as sm\n",
    "from scipy.stats import t\n",
    "nb = MultinomialNB()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def efficiency(data_dairy):\n",
    "    eff_calc = (data_dairy/101.336)*100\n",
    "    return  eff_calc"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The daily biogas generation is a number that is reported to the EPA by the farm; it is not a calculated number. Digester owners could misreport their daily biogas production, thereby skewing the data. Later, you will see from the histograms that the digester daily biogas per animal distributions are almost all right skewed (a few over-optimistic owners). Therefore, to better analyze the data, I use the two functions below to disclude data that falls outside of 1 or 2 standard deviations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "#95% confidence interval filter for data - 2 standard deviations\n",
    "\n",
    "def hist_filter_ci(data):\n",
    "    Y_upper = np.percentile(data['Biogas_ft3/cow'], 97.5)\n",
    "    Y_lower = np.percentile(data['Biogas_ft3/cow'], 2.5)\n",
    "    filtered_hist_data = data[(data['Biogas_ft3/cow'] >= Y_lower) & (data['Biogas_ft3/cow'] <= Y_upper)]\n",
    "    return filtered_hist_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#68% confidence interval filter for data- 1 standard deviation\n",
    "\n",
    "def hist_filter_ci_68(data):\n",
    "    Y_upper = np.percentile(data['Biogas_ft3/cow'], 84)\n",
    "    Y_lower = np.percentile(data['Biogas_ft3/cow'], 16)\n",
    "    filtered_hist_data = data[(data['Biogas_ft3/cow'] >= Y_lower) & (data['Biogas_ft3/cow'] <= Y_upper)]\n",
    "    return filtered_hist_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = agstar.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.fillna(0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Project Name</th>\n",
       "      <th>Project Type</th>\n",
       "      <th>City</th>\n",
       "      <th>County</th>\n",
       "      <th>State</th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Status</th>\n",
       "      <th>Year Operational</th>\n",
       "      <th>Animal/Farm Type(s)</th>\n",
       "      <th>Cattle</th>\n",
       "      <th>...</th>\n",
       "      <th>Poultry</th>\n",
       "      <th>Swine</th>\n",
       "      <th>Co-Digestion</th>\n",
       "      <th>Biogas Generation Estimate (cu_ft/day)</th>\n",
       "      <th>Electricity Generated (kWh/yr)</th>\n",
       "      <th>Biogas End Use(s)</th>\n",
       "      <th>System Designer(s)_Developer(s) and Affiliates</th>\n",
       "      <th>Receiving Utility</th>\n",
       "      <th>Total Emission Reductions (MTCO2e/yr)</th>\n",
       "      <th>Awarded USDA Funding?</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Cargill - Sandy River Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Morrilton</td>\n",
       "      <td>Conway</td>\n",
       "      <td>AR</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>Swine</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>4,200</td>\n",
       "      <td>0</td>\n",
       "      <td>1,814,400</td>\n",
       "      <td>0</td>\n",
       "      <td>Flared Full-time</td>\n",
       "      <td>Martin Construction Resource LLC (formerly RCM...</td>\n",
       "      <td>0</td>\n",
       "      <td>4,002</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Butterfield RNG Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Buckeye</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Mixed Plug Flow</td>\n",
       "      <td>Construction</td>\n",
       "      <td>2021.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Pipeline Gas</td>\n",
       "      <td>Avolta [Project Developer]; DVO, Inc. (formerl...</td>\n",
       "      <td>Southwest Gas</td>\n",
       "      <td>29,826</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Caballero Dairy Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Eloy</td>\n",
       "      <td>Pinal</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Unknown or Unspecified</td>\n",
       "      <td>Construction</td>\n",
       "      <td>2022.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Pipeline Gas</td>\n",
       "      <td>Brightmark [Project Developer]</td>\n",
       "      <td>0</td>\n",
       "      <td>89,518</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Paloma Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Gila Bend</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2021.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>CNG</td>\n",
       "      <td>Black Bear Environmental Assets [Project Devel...</td>\n",
       "      <td>Southwest Gas Company</td>\n",
       "      <td>89,794</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Stotz Southern Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Buckeye</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2011.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5,256,000</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Chapel Street Environmental [System Design Eng...</td>\n",
       "      <td>Arizona Public Service</td>\n",
       "      <td>138,787</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>501</th>\n",
       "      <td>Norswiss Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Rice Lake</td>\n",
       "      <td>Barron</td>\n",
       "      <td>WI</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2006.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Dairy Processing Wastes; Fats, Oils, Greases; ...</td>\n",
       "      <td>2,908,000</td>\n",
       "      <td>6,450,000</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Microgy [Project Developer, System Designer]</td>\n",
       "      <td>Dairyland Power Cooperative; Barron Electric</td>\n",
       "      <td>0</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>502</th>\n",
       "      <td>Quantum Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Weyauwega</td>\n",
       "      <td>Waupaca</td>\n",
       "      <td>WI</td>\n",
       "      <td>Mixed Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2005.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3,350,700</td>\n",
       "      <td>Cogeneration</td>\n",
       "      <td>DVO, Inc. (formerly GHD, Inc.) [Project Develo...</td>\n",
       "      <td>WE Energies, Inc.</td>\n",
       "      <td>0</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>503</th>\n",
       "      <td>Stencil Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Denmark</td>\n",
       "      <td>Brown</td>\n",
       "      <td>WI</td>\n",
       "      <td>Horizontal Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2002.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>946,080</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Martin Construction Resource LLC (formerly RCM...</td>\n",
       "      <td>Wisconsin Public Service Corporation</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504</th>\n",
       "      <td>Tinedale Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Wrightstown</td>\n",
       "      <td>Jackson</td>\n",
       "      <td>WI</td>\n",
       "      <td>Fixed Film/Attached Media</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>1999.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>200,000</td>\n",
       "      <td>5,584,500</td>\n",
       "      <td>Electricity; Boiler/Furnace fuel</td>\n",
       "      <td>AGES [Project Developer, System Designer, Syst...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>505</th>\n",
       "      <td>Wyoming Premium Farms 2 Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Wheatland</td>\n",
       "      <td>Platte</td>\n",
       "      <td>WY</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2004.0</td>\n",
       "      <td>Swine</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>18,000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1,340,280</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>AG Engineering, Inc. [System Design Engineer];...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>506 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                            Project Name Project Type         City    County  \\\n",
       "0    Cargill - Sandy River Farm Digester   Farm Scale    Morrilton    Conway   \n",
       "1               Butterfield RNG Digester   Farm Scale      Buckeye  Maricopa   \n",
       "2         Caballero Dairy Farms Digester   Farm Scale         Eloy     Pinal   \n",
       "3                  Paloma Dairy Digester   Farm Scale    Gila Bend  Maricopa   \n",
       "4          Stotz Southern Dairy Digester   Farm Scale      Buckeye  Maricopa   \n",
       "..                                   ...          ...          ...       ...   \n",
       "501              Norswiss Farms Digester   Farm Scale    Rice Lake    Barron   \n",
       "502               Quantum Dairy Digester   Farm Scale    Weyauwega   Waupaca   \n",
       "503                Stencil Farm Digester   Farm Scale      Denmark     Brown   \n",
       "504              Tinedale Farms Digester   Farm Scale  Wrightstown   Jackson   \n",
       "505     Wyoming Premium Farms 2 Digester   Farm Scale    Wheatland    Platte   \n",
       "\n",
       "    State              Digester Type        Status  Year Operational  \\\n",
       "0      AR             Covered Lagoon   Operational            2008.0   \n",
       "1      AZ            Mixed Plug Flow  Construction            2021.0   \n",
       "2      AZ     Unknown or Unspecified  Construction            2022.0   \n",
       "3      AZ               Complete Mix   Operational            2021.0   \n",
       "4      AZ             Covered Lagoon   Operational            2011.0   \n",
       "..    ...                        ...           ...               ...   \n",
       "501    WI               Complete Mix     Shut down            2006.0   \n",
       "502    WI            Mixed Plug Flow     Shut down            2005.0   \n",
       "503    WI       Horizontal Plug Flow     Shut down            2002.0   \n",
       "504    WI  Fixed Film/Attached Media     Shut down            1999.0   \n",
       "505    WY               Complete Mix     Shut down            2004.0   \n",
       "\n",
       "    Animal/Farm Type(s) Cattle  ... Poultry   Swine  \\\n",
       "0                 Swine      0  ...       0   4,200   \n",
       "1                 Dairy      0  ...       0       0   \n",
       "2                 Dairy      0  ...       0       0   \n",
       "3                 Dairy      0  ...       0       0   \n",
       "4                 Dairy      0  ...       0       0   \n",
       "..                  ...    ...  ...     ...     ...   \n",
       "501               Dairy      0  ...       0       0   \n",
       "502               Dairy      0  ...       0       0   \n",
       "503               Dairy      0  ...       0       0   \n",
       "504               Dairy      0  ...       0       0   \n",
       "505               Swine      0  ...       0  18,000   \n",
       "\n",
       "                                          Co-Digestion  \\\n",
       "0                                                    0   \n",
       "1                                                    0   \n",
       "2                                                    0   \n",
       "3                                                    0   \n",
       "4                                                    0   \n",
       "..                                                 ...   \n",
       "501  Dairy Processing Wastes; Fats, Oils, Greases; ...   \n",
       "502                                                  0   \n",
       "503                                                  0   \n",
       "504                                                  0   \n",
       "505                                                  0   \n",
       "\n",
       "     Biogas Generation Estimate (cu_ft/day)   Electricity Generated (kWh/yr)   \\\n",
       "0                                  1,814,400                                0   \n",
       "1                                          0                                0   \n",
       "2                                          0                                0   \n",
       "3                                          0                                0   \n",
       "4                                          0                        5,256,000   \n",
       "..                                       ...                              ...   \n",
       "501                                2,908,000                        6,450,000   \n",
       "502                                        0                        3,350,700   \n",
       "503                                        0                          946,080   \n",
       "504                                  200,000                        5,584,500   \n",
       "505                                        0                        1,340,280   \n",
       "\n",
       "                    Biogas End Use(s)  \\\n",
       "0                    Flared Full-time   \n",
       "1                        Pipeline Gas   \n",
       "2                        Pipeline Gas   \n",
       "3                                 CNG   \n",
       "4                         Electricity   \n",
       "..                                ...   \n",
       "501                       Electricity   \n",
       "502                      Cogeneration   \n",
       "503                       Electricity   \n",
       "504  Electricity; Boiler/Furnace fuel   \n",
       "505                       Electricity   \n",
       "\n",
       "        System Designer(s)_Developer(s) and Affiliates  \\\n",
       "0    Martin Construction Resource LLC (formerly RCM...   \n",
       "1    Avolta [Project Developer]; DVO, Inc. (formerl...   \n",
       "2                       Brightmark [Project Developer]   \n",
       "3    Black Bear Environmental Assets [Project Devel...   \n",
       "4    Chapel Street Environmental [System Design Eng...   \n",
       "..                                                 ...   \n",
       "501       Microgy [Project Developer, System Designer]   \n",
       "502  DVO, Inc. (formerly GHD, Inc.) [Project Develo...   \n",
       "503  Martin Construction Resource LLC (formerly RCM...   \n",
       "504  AGES [Project Developer, System Designer, Syst...   \n",
       "505  AG Engineering, Inc. [System Design Engineer];...   \n",
       "\n",
       "                                Receiving Utility  \\\n",
       "0                                               0   \n",
       "1                                   Southwest Gas   \n",
       "2                                               0   \n",
       "3                           Southwest Gas Company   \n",
       "4                          Arizona Public Service   \n",
       "..                                            ...   \n",
       "501  Dairyland Power Cooperative; Barron Electric   \n",
       "502                             WE Energies, Inc.   \n",
       "503          Wisconsin Public Service Corporation   \n",
       "504                                             0   \n",
       "505                                             0   \n",
       "\n",
       "     Total Emission Reductions (MTCO2e/yr)  Awarded USDA Funding?  \n",
       "0                                     4,002                     0  \n",
       "1                                    29,826                     Y  \n",
       "2                                    89,518                     0  \n",
       "3                                    89,794                     0  \n",
       "4                                   138,787                     0  \n",
       "..                                      ...                   ...  \n",
       "501                                       0                     Y  \n",
       "502                                       0                     Y  \n",
       "503                                       0                     0  \n",
       "504                                       0                     0  \n",
       "505                                       0                     0  \n",
       "\n",
       "[506 rows x 21 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "#remove commas within numbers in dataset then assign numbers the type \"integer\"\n",
    "\n",
    "df[' Biogas Generation Estimate (cu_ft/day) ']=df[' Biogas Generation Estimate (cu_ft/day) '].str.replace(',','')\n",
    "df[' Electricity Generated (kWh/yr) ']=df[' Electricity Generated (kWh/yr) '].str.replace(',','')\n",
    "df['Dairy']=df['Dairy'].str.replace(',','')\n",
    "df['Swine']=df['Swine'].str.replace(',','')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#fill na with 0\n",
    "df.fillna(0,inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.astype({' Biogas Generation Estimate (cu_ft/day) ':'int'})\n",
    "df = df.astype({' Electricity Generated (kWh/yr) ': 'int'})\n",
    "df = df.astype({'Dairy':'int'})\n",
    "df = df.astype({'Swine':'int'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Project Name</th>\n",
       "      <th>Project Type</th>\n",
       "      <th>City</th>\n",
       "      <th>County</th>\n",
       "      <th>State</th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Status</th>\n",
       "      <th>Year Operational</th>\n",
       "      <th>Animal/Farm Type(s)</th>\n",
       "      <th>Cattle</th>\n",
       "      <th>...</th>\n",
       "      <th>Poultry</th>\n",
       "      <th>Swine</th>\n",
       "      <th>Co-Digestion</th>\n",
       "      <th>Biogas Generation Estimate (cu_ft/day)</th>\n",
       "      <th>Electricity Generated (kWh/yr)</th>\n",
       "      <th>Biogas End Use(s)</th>\n",
       "      <th>System Designer(s)_Developer(s) and Affiliates</th>\n",
       "      <th>Receiving Utility</th>\n",
       "      <th>Total Emission Reductions (MTCO2e/yr)</th>\n",
       "      <th>Awarded USDA Funding?</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Cargill - Sandy River Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Morrilton</td>\n",
       "      <td>Conway</td>\n",
       "      <td>AR</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>Swine</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>4200</td>\n",
       "      <td>0</td>\n",
       "      <td>1814400</td>\n",
       "      <td>0</td>\n",
       "      <td>Flared Full-time</td>\n",
       "      <td>Martin Construction Resource LLC (formerly RCM...</td>\n",
       "      <td>0</td>\n",
       "      <td>4,002</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Butterfield RNG Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Buckeye</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Mixed Plug Flow</td>\n",
       "      <td>Construction</td>\n",
       "      <td>2021.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Pipeline Gas</td>\n",
       "      <td>Avolta [Project Developer]; DVO, Inc. (formerl...</td>\n",
       "      <td>Southwest Gas</td>\n",
       "      <td>29,826</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Caballero Dairy Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Eloy</td>\n",
       "      <td>Pinal</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Unknown or Unspecified</td>\n",
       "      <td>Construction</td>\n",
       "      <td>2022.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Pipeline Gas</td>\n",
       "      <td>Brightmark [Project Developer]</td>\n",
       "      <td>0</td>\n",
       "      <td>89,518</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Paloma Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Gila Bend</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2021.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>CNG</td>\n",
       "      <td>Black Bear Environmental Assets [Project Devel...</td>\n",
       "      <td>Southwest Gas Company</td>\n",
       "      <td>89,794</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Stotz Southern Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Buckeye</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2011.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5256000</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Chapel Street Environmental [System Design Eng...</td>\n",
       "      <td>Arizona Public Service</td>\n",
       "      <td>138,787</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                          Project Name Project Type       City    County  \\\n",
       "0  Cargill - Sandy River Farm Digester   Farm Scale  Morrilton    Conway   \n",
       "1             Butterfield RNG Digester   Farm Scale    Buckeye  Maricopa   \n",
       "2       Caballero Dairy Farms Digester   Farm Scale       Eloy     Pinal   \n",
       "3                Paloma Dairy Digester   Farm Scale  Gila Bend  Maricopa   \n",
       "4        Stotz Southern Dairy Digester   Farm Scale    Buckeye  Maricopa   \n",
       "\n",
       "  State           Digester Type        Status  Year Operational  \\\n",
       "0    AR          Covered Lagoon   Operational            2008.0   \n",
       "1    AZ         Mixed Plug Flow  Construction            2021.0   \n",
       "2    AZ  Unknown or Unspecified  Construction            2022.0   \n",
       "3    AZ            Complete Mix   Operational            2021.0   \n",
       "4    AZ          Covered Lagoon   Operational            2011.0   \n",
       "\n",
       "  Animal/Farm Type(s) Cattle  ...  Poultry Swine  Co-Digestion  \\\n",
       "0               Swine      0  ...        0  4200             0   \n",
       "1               Dairy      0  ...        0     0             0   \n",
       "2               Dairy      0  ...        0     0             0   \n",
       "3               Dairy      0  ...        0     0             0   \n",
       "4               Dairy      0  ...        0     0             0   \n",
       "\n",
       "   Biogas Generation Estimate (cu_ft/day)    Electricity Generated (kWh/yr)   \\\n",
       "0                                  1814400                                 0   \n",
       "1                                        0                                 0   \n",
       "2                                        0                                 0   \n",
       "3                                        0                                 0   \n",
       "4                                        0                           5256000   \n",
       "\n",
       "   Biogas End Use(s)     System Designer(s)_Developer(s) and Affiliates  \\\n",
       "0   Flared Full-time  Martin Construction Resource LLC (formerly RCM...   \n",
       "1       Pipeline Gas  Avolta [Project Developer]; DVO, Inc. (formerl...   \n",
       "2       Pipeline Gas                     Brightmark [Project Developer]   \n",
       "3                CNG  Black Bear Environmental Assets [Project Devel...   \n",
       "4        Electricity  Chapel Street Environmental [System Design Eng...   \n",
       "\n",
       "        Receiving Utility  Total Emission Reductions (MTCO2e/yr)   \\\n",
       "0                       0                                   4,002   \n",
       "1           Southwest Gas                                  29,826   \n",
       "2                       0                                  89,518   \n",
       "3   Southwest Gas Company                                  89,794   \n",
       "4  Arizona Public Service                                 138,787   \n",
       "\n",
       "  Awarded USDA Funding?  \n",
       "0                     0  \n",
       "1                     Y  \n",
       "2                     0  \n",
       "3                     0  \n",
       "4                     0  \n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2 = df.rename(columns={\"Animal/Farm Type(s)\" : \"Animal\", \"Co-Digestion\" : \"Codigestion\", \"Biogas End Use(s)\" : \"Biogas_End_Use\", \" Biogas Generation Estimate (cu_ft/day) \" : \"Biogas_gen_ft3_day\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Project Name', 'Project Type', 'City', 'County', 'State',\n",
       "       'Digester Type', 'Status', 'Year Operational', 'Animal', 'Cattle',\n",
       "       'Dairy', 'Poultry', 'Swine', 'Codigestion', 'Biogas_gen_ft3_day',\n",
       "       ' Electricity Generated (kWh/yr) ', 'Biogas_End_Use',\n",
       "       'System Designer(s)_Developer(s) and Affiliates', 'Receiving Utility',\n",
       "       ' Total Emission Reductions (MTCO2e/yr) ', 'Awarded USDA Funding?'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.columns "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Project Name</th>\n",
       "      <th>Project Type</th>\n",
       "      <th>City</th>\n",
       "      <th>County</th>\n",
       "      <th>State</th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Status</th>\n",
       "      <th>Year Operational</th>\n",
       "      <th>Animal</th>\n",
       "      <th>Cattle</th>\n",
       "      <th>...</th>\n",
       "      <th>Poultry</th>\n",
       "      <th>Swine</th>\n",
       "      <th>Codigestion</th>\n",
       "      <th>Biogas_gen_ft3_day</th>\n",
       "      <th>Electricity Generated (kWh/yr)</th>\n",
       "      <th>Biogas_End_Use</th>\n",
       "      <th>System Designer(s)_Developer(s) and Affiliates</th>\n",
       "      <th>Receiving Utility</th>\n",
       "      <th>Total Emission Reductions (MTCO2e/yr)</th>\n",
       "      <th>Awarded USDA Funding?</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Cargill - Sandy River Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Morrilton</td>\n",
       "      <td>Conway</td>\n",
       "      <td>AR</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>Swine</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>4200</td>\n",
       "      <td>0</td>\n",
       "      <td>1814400</td>\n",
       "      <td>0</td>\n",
       "      <td>Flared Full-time</td>\n",
       "      <td>Martin Construction Resource LLC (formerly RCM...</td>\n",
       "      <td>0</td>\n",
       "      <td>4,002</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Butterfield RNG Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Buckeye</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Mixed Plug Flow</td>\n",
       "      <td>Construction</td>\n",
       "      <td>2021.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Pipeline Gas</td>\n",
       "      <td>Avolta [Project Developer]; DVO, Inc. (formerl...</td>\n",
       "      <td>Southwest Gas</td>\n",
       "      <td>29,826</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Caballero Dairy Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Eloy</td>\n",
       "      <td>Pinal</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Unknown or Unspecified</td>\n",
       "      <td>Construction</td>\n",
       "      <td>2022.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Pipeline Gas</td>\n",
       "      <td>Brightmark [Project Developer]</td>\n",
       "      <td>0</td>\n",
       "      <td>89,518</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Paloma Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Gila Bend</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2021.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>CNG</td>\n",
       "      <td>Black Bear Environmental Assets [Project Devel...</td>\n",
       "      <td>Southwest Gas Company</td>\n",
       "      <td>89,794</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Stotz Southern Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Buckeye</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2011.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5256000</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Chapel Street Environmental [System Design Eng...</td>\n",
       "      <td>Arizona Public Service</td>\n",
       "      <td>138,787</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>501</th>\n",
       "      <td>Norswiss Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Rice Lake</td>\n",
       "      <td>Barron</td>\n",
       "      <td>WI</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2006.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Dairy Processing Wastes; Fats, Oils, Greases; ...</td>\n",
       "      <td>2908000</td>\n",
       "      <td>6450000</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Microgy [Project Developer, System Designer]</td>\n",
       "      <td>Dairyland Power Cooperative; Barron Electric</td>\n",
       "      <td>0</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>502</th>\n",
       "      <td>Quantum Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Weyauwega</td>\n",
       "      <td>Waupaca</td>\n",
       "      <td>WI</td>\n",
       "      <td>Mixed Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2005.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3350700</td>\n",
       "      <td>Cogeneration</td>\n",
       "      <td>DVO, Inc. (formerly GHD, Inc.) [Project Develo...</td>\n",
       "      <td>WE Energies, Inc.</td>\n",
       "      <td>0</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>503</th>\n",
       "      <td>Stencil Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Denmark</td>\n",
       "      <td>Brown</td>\n",
       "      <td>WI</td>\n",
       "      <td>Horizontal Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2002.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>946080</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Martin Construction Resource LLC (formerly RCM...</td>\n",
       "      <td>Wisconsin Public Service Corporation</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504</th>\n",
       "      <td>Tinedale Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Wrightstown</td>\n",
       "      <td>Jackson</td>\n",
       "      <td>WI</td>\n",
       "      <td>Fixed Film/Attached Media</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>1999.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>200000</td>\n",
       "      <td>5584500</td>\n",
       "      <td>Electricity; Boiler/Furnace fuel</td>\n",
       "      <td>AGES [Project Developer, System Designer, Syst...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>505</th>\n",
       "      <td>Wyoming Premium Farms 2 Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Wheatland</td>\n",
       "      <td>Platte</td>\n",
       "      <td>WY</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2004.0</td>\n",
       "      <td>Swine</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>18000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1340280</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>AG Engineering, Inc. [System Design Engineer];...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>506 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                            Project Name Project Type         City    County  \\\n",
       "0    Cargill - Sandy River Farm Digester   Farm Scale    Morrilton    Conway   \n",
       "1               Butterfield RNG Digester   Farm Scale      Buckeye  Maricopa   \n",
       "2         Caballero Dairy Farms Digester   Farm Scale         Eloy     Pinal   \n",
       "3                  Paloma Dairy Digester   Farm Scale    Gila Bend  Maricopa   \n",
       "4          Stotz Southern Dairy Digester   Farm Scale      Buckeye  Maricopa   \n",
       "..                                   ...          ...          ...       ...   \n",
       "501              Norswiss Farms Digester   Farm Scale    Rice Lake    Barron   \n",
       "502               Quantum Dairy Digester   Farm Scale    Weyauwega   Waupaca   \n",
       "503                Stencil Farm Digester   Farm Scale      Denmark     Brown   \n",
       "504              Tinedale Farms Digester   Farm Scale  Wrightstown   Jackson   \n",
       "505     Wyoming Premium Farms 2 Digester   Farm Scale    Wheatland    Platte   \n",
       "\n",
       "    State              Digester Type        Status  Year Operational Animal  \\\n",
       "0      AR             Covered Lagoon   Operational            2008.0  Swine   \n",
       "1      AZ            Mixed Plug Flow  Construction            2021.0  Dairy   \n",
       "2      AZ     Unknown or Unspecified  Construction            2022.0  Dairy   \n",
       "3      AZ               Complete Mix   Operational            2021.0  Dairy   \n",
       "4      AZ             Covered Lagoon   Operational            2011.0  Dairy   \n",
       "..    ...                        ...           ...               ...    ...   \n",
       "501    WI               Complete Mix     Shut down            2006.0  Dairy   \n",
       "502    WI            Mixed Plug Flow     Shut down            2005.0  Dairy   \n",
       "503    WI       Horizontal Plug Flow     Shut down            2002.0  Dairy   \n",
       "504    WI  Fixed Film/Attached Media     Shut down            1999.0  Dairy   \n",
       "505    WY               Complete Mix     Shut down            2004.0  Swine   \n",
       "\n",
       "    Cattle  ...  Poultry  Swine  \\\n",
       "0        0  ...        0   4200   \n",
       "1        0  ...        0      0   \n",
       "2        0  ...        0      0   \n",
       "3        0  ...        0      0   \n",
       "4        0  ...        0      0   \n",
       "..     ...  ...      ...    ...   \n",
       "501      0  ...        0      0   \n",
       "502      0  ...        0      0   \n",
       "503      0  ...        0      0   \n",
       "504      0  ...        0      0   \n",
       "505      0  ...        0  18000   \n",
       "\n",
       "                                           Codigestion Biogas_gen_ft3_day  \\\n",
       "0                                                    0            1814400   \n",
       "1                                                    0                  0   \n",
       "2                                                    0                  0   \n",
       "3                                                    0                  0   \n",
       "4                                                    0                  0   \n",
       "..                                                 ...                ...   \n",
       "501  Dairy Processing Wastes; Fats, Oils, Greases; ...            2908000   \n",
       "502                                                  0                  0   \n",
       "503                                                  0                  0   \n",
       "504                                                  0             200000   \n",
       "505                                                  0                  0   \n",
       "\n",
       "      Electricity Generated (kWh/yr)                     Biogas_End_Use  \\\n",
       "0                                   0                  Flared Full-time   \n",
       "1                                   0                      Pipeline Gas   \n",
       "2                                   0                      Pipeline Gas   \n",
       "3                                   0                               CNG   \n",
       "4                             5256000                       Electricity   \n",
       "..                                ...                               ...   \n",
       "501                           6450000                       Electricity   \n",
       "502                           3350700                      Cogeneration   \n",
       "503                            946080                       Electricity   \n",
       "504                           5584500  Electricity; Boiler/Furnace fuel   \n",
       "505                           1340280                       Electricity   \n",
       "\n",
       "        System Designer(s)_Developer(s) and Affiliates  \\\n",
       "0    Martin Construction Resource LLC (formerly RCM...   \n",
       "1    Avolta [Project Developer]; DVO, Inc. (formerl...   \n",
       "2                       Brightmark [Project Developer]   \n",
       "3    Black Bear Environmental Assets [Project Devel...   \n",
       "4    Chapel Street Environmental [System Design Eng...   \n",
       "..                                                 ...   \n",
       "501       Microgy [Project Developer, System Designer]   \n",
       "502  DVO, Inc. (formerly GHD, Inc.) [Project Develo...   \n",
       "503  Martin Construction Resource LLC (formerly RCM...   \n",
       "504  AGES [Project Developer, System Designer, Syst...   \n",
       "505  AG Engineering, Inc. [System Design Engineer];...   \n",
       "\n",
       "                                Receiving Utility  \\\n",
       "0                                               0   \n",
       "1                                   Southwest Gas   \n",
       "2                                               0   \n",
       "3                           Southwest Gas Company   \n",
       "4                          Arizona Public Service   \n",
       "..                                            ...   \n",
       "501  Dairyland Power Cooperative; Barron Electric   \n",
       "502                             WE Energies, Inc.   \n",
       "503          Wisconsin Public Service Corporation   \n",
       "504                                             0   \n",
       "505                                             0   \n",
       "\n",
       "     Total Emission Reductions (MTCO2e/yr)  Awarded USDA Funding?  \n",
       "0                                     4,002                     0  \n",
       "1                                    29,826                     Y  \n",
       "2                                    89,518                     0  \n",
       "3                                    89,794                     0  \n",
       "4                                   138,787                     0  \n",
       "..                                      ...                   ...  \n",
       "501                                       0                     Y  \n",
       "502                                       0                     Y  \n",
       "503                                       0                     0  \n",
       "504                                       0                     0  \n",
       "505                                       0                     0  \n",
       "\n",
       "[506 rows x 21 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Throughout the notebook, I drop rows in dataframes for various reasons. Most of the time I'm discluding '0' values or selecting the animal I want to analyze. \n",
    "\n",
    "Also, I never want data where there is \"codigestion\" involved (mutliple feedstocks entering the digester). Typically, codigestion means there is an agricultural residue (corn stover, wheat stalks, brewers grains, etc) added along with manure to the digester. My maximum biogas per animal values, which are used to calculate digester efficiency, are based on a manure-only operating digester.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop unwanted rows from Dataframe\n",
    "\n",
    "df2.drop(df2[(df2['Animal'] != 'Dairy')].index, inplace = True)\n",
    "df2.drop(df2[(df2['Codigestion'] != 0)].index, inplace = True)\n",
    "df2.drop(df2[(df2['Biogas_gen_ft3_day'] == 0)].index, inplace = True)\n",
    "df2.drop(df2[(df2['Biogas_End_Use'] == 0)].index, inplace = True)\n",
    "\n",
    "df2['Biogas_ft3/cow'] = df2['Biogas_gen_ft3_day'] / df2['Dairy']\n",
    "\n",
    "#notwant = ['Electricity', 'Cogeneration; Refrigeration', 'Electricity; CNG',\n",
    "       #'Boiler/Furnace fuel', 'Electricity; Boiler/Furnace fuel',\n",
    "       #'Cogeneration', 'Cogeneration; CNG', 'Electricity; Cogeneration',\n",
    "       #'Flared Full-time', 'Cogeneration; Boiler/Furnace fuel', 'Electricity; Pipeline Gas',\n",
    "       #'Cogeneration; Pipeline Gas']\n",
    "\n",
    "#df2 = df2[~df2['Biogas_End_Use'].isin(notwant)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Project Name</th>\n",
       "      <th>Project Type</th>\n",
       "      <th>City</th>\n",
       "      <th>County</th>\n",
       "      <th>State</th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Status</th>\n",
       "      <th>Year Operational</th>\n",
       "      <th>Animal</th>\n",
       "      <th>Cattle</th>\n",
       "      <th>...</th>\n",
       "      <th>Swine</th>\n",
       "      <th>Codigestion</th>\n",
       "      <th>Biogas_gen_ft3_day</th>\n",
       "      <th>Electricity Generated (kWh/yr)</th>\n",
       "      <th>Biogas_End_Use</th>\n",
       "      <th>System Designer(s)_Developer(s) and Affiliates</th>\n",
       "      <th>Receiving Utility</th>\n",
       "      <th>Total Emission Reductions (MTCO2e/yr)</th>\n",
       "      <th>Awarded USDA Funding?</th>\n",
       "      <th>Biogas_ft3/cow</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>ABEC Bidart-Old River LLC Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Bakersfield</td>\n",
       "      <td>Kern</td>\n",
       "      <td>CA</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2013.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>600000</td>\n",
       "      <td>16206000</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>4Creeks [System Designer, System Design Engine...</td>\n",
       "      <td>PG&amp;E</td>\n",
       "      <td>107,483</td>\n",
       "      <td>0</td>\n",
       "      <td>38.709677</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>ABEC Bidart-Stockdale LLC Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Bakersfield</td>\n",
       "      <td>Kern</td>\n",
       "      <td>CA</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2013.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>50000</td>\n",
       "      <td>4204800</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>California Bioenergy LLC [Project Developer, S...</td>\n",
       "      <td>PG&amp;E</td>\n",
       "      <td>14,867</td>\n",
       "      <td>0</td>\n",
       "      <td>29.411765</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>ABEC Carlos Echeverria &amp; Sons Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Bakersfield</td>\n",
       "      <td>Kern</td>\n",
       "      <td>CA</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2018.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>270000</td>\n",
       "      <td>7600000</td>\n",
       "      <td>Cogeneration; Refrigeration</td>\n",
       "      <td>California Bioenergy LLC [Project Developer]; ...</td>\n",
       "      <td>0</td>\n",
       "      <td>82,427</td>\n",
       "      <td>0</td>\n",
       "      <td>27.835052</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>ABEC Lakeview Farms Dairy Digester</td>\n",
       "      <td>Multiple Farm/Facility</td>\n",
       "      <td>Bakersfield</td>\n",
       "      <td>Kern</td>\n",
       "      <td>CA</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2018.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>360000</td>\n",
       "      <td>6700000</td>\n",
       "      <td>Electricity; CNG</td>\n",
       "      <td>California Bioenergy LLC [Project Developer]</td>\n",
       "      <td>Pacific Gas and Electric Company</td>\n",
       "      <td>60,350</td>\n",
       "      <td>0</td>\n",
       "      <td>51.428571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>Castelanelli Bros. Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Lodi</td>\n",
       "      <td>San Joaquin</td>\n",
       "      <td>CA</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2004.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>89148</td>\n",
       "      <td>2233800</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Environmental Fabrics, Inc. [Biogas Membrane S...</td>\n",
       "      <td>Pacific Gas and Electric Company</td>\n",
       "      <td>19,462</td>\n",
       "      <td>Y</td>\n",
       "      <td>27.746032</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>489</th>\n",
       "      <td>Foote Farm Digester</td>\n",
       "      <td>Research</td>\n",
       "      <td>Charlotte</td>\n",
       "      <td>Chittenden</td>\n",
       "      <td>VT</td>\n",
       "      <td>Modular Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2005.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5000</td>\n",
       "      <td>148920</td>\n",
       "      <td>Cogeneration</td>\n",
       "      <td>Avatar Energy [Project Developer, System Desig...</td>\n",
       "      <td>Green Mountain Power</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>31.250000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>490</th>\n",
       "      <td>Foster Brothers Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Middlebury</td>\n",
       "      <td>Addison</td>\n",
       "      <td>VT</td>\n",
       "      <td>Horizontal Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>1982.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>37500</td>\n",
       "      <td>480000</td>\n",
       "      <td>Electricity; Cogeneration</td>\n",
       "      <td>Hadley and Bennett [System Designer, System De...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>98.684211</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>491</th>\n",
       "      <td>Joneslan Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Hyde Park</td>\n",
       "      <td>Lamoille</td>\n",
       "      <td>VT</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2012.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>33450</td>\n",
       "      <td>223380</td>\n",
       "      <td>Cogeneration</td>\n",
       "      <td>UEM Group [System Designer, System Design Engi...</td>\n",
       "      <td>Vermont Electric Cooperative</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>95.571429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>492</th>\n",
       "      <td>Keewaydin Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Stowe</td>\n",
       "      <td>Lamoille</td>\n",
       "      <td>VT</td>\n",
       "      <td>Modular Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2011.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5000</td>\n",
       "      <td>87600</td>\n",
       "      <td>Cogeneration</td>\n",
       "      <td>Avatar Energy [Project Developer, System Desig...</td>\n",
       "      <td>Stowe Electric</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>41.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504</th>\n",
       "      <td>Tinedale Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Wrightstown</td>\n",
       "      <td>Jackson</td>\n",
       "      <td>WI</td>\n",
       "      <td>Fixed Film/Attached Media</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>1999.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>200000</td>\n",
       "      <td>5584500</td>\n",
       "      <td>Electricity; Boiler/Furnace fuel</td>\n",
       "      <td>AGES [Project Developer, System Designer, Syst...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111.111111</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>67 rows × 22 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                               Project Name            Project Type  \\\n",
       "7        ABEC Bidart-Old River LLC Digester              Farm Scale   \n",
       "8        ABEC Bidart-Stockdale LLC Digester              Farm Scale   \n",
       "9    ABEC Carlos Echeverria & Sons Digester              Farm Scale   \n",
       "10       ABEC Lakeview Farms Dairy Digester  Multiple Farm/Facility   \n",
       "27        Castelanelli Bros. Dairy Digester              Farm Scale   \n",
       "..                                      ...                     ...   \n",
       "489                     Foote Farm Digester                Research   \n",
       "490          Foster Brothers Farms Digester              Farm Scale   \n",
       "491                  Joneslan Farm Digester              Farm Scale   \n",
       "492                 Keewaydin Farm Digester              Farm Scale   \n",
       "504                 Tinedale Farms Digester              Farm Scale   \n",
       "\n",
       "            City       County State              Digester Type       Status  \\\n",
       "7    Bakersfield         Kern    CA             Covered Lagoon  Operational   \n",
       "8    Bakersfield         Kern    CA             Covered Lagoon  Operational   \n",
       "9    Bakersfield         Kern    CA             Covered Lagoon  Operational   \n",
       "10   Bakersfield         Kern    CA             Covered Lagoon  Operational   \n",
       "27          Lodi  San Joaquin    CA             Covered Lagoon  Operational   \n",
       "..           ...          ...   ...                        ...          ...   \n",
       "489    Charlotte   Chittenden    VT          Modular Plug Flow    Shut down   \n",
       "490   Middlebury      Addison    VT       Horizontal Plug Flow    Shut down   \n",
       "491    Hyde Park     Lamoille    VT               Complete Mix    Shut down   \n",
       "492        Stowe     Lamoille    VT          Modular Plug Flow    Shut down   \n",
       "504  Wrightstown      Jackson    WI  Fixed Film/Attached Media    Shut down   \n",
       "\n",
       "     Year Operational Animal Cattle  ...  Swine Codigestion  \\\n",
       "7              2013.0  Dairy      0  ...      0           0   \n",
       "8              2013.0  Dairy      0  ...      0           0   \n",
       "9              2018.0  Dairy      0  ...      0           0   \n",
       "10             2018.0  Dairy      0  ...      0           0   \n",
       "27             2004.0  Dairy      0  ...      0           0   \n",
       "..                ...    ...    ...  ...    ...         ...   \n",
       "489            2005.0  Dairy      0  ...      0           0   \n",
       "490            1982.0  Dairy      0  ...      0           0   \n",
       "491            2012.0  Dairy      0  ...      0           0   \n",
       "492            2011.0  Dairy      0  ...      0           0   \n",
       "504            1999.0  Dairy      0  ...      0           0   \n",
       "\n",
       "     Biogas_gen_ft3_day  Electricity Generated (kWh/yr)   \\\n",
       "7                600000                         16206000   \n",
       "8                 50000                          4204800   \n",
       "9                270000                          7600000   \n",
       "10               360000                          6700000   \n",
       "27                89148                          2233800   \n",
       "..                  ...                              ...   \n",
       "489                5000                           148920   \n",
       "490               37500                           480000   \n",
       "491               33450                           223380   \n",
       "492                5000                            87600   \n",
       "504              200000                          5584500   \n",
       "\n",
       "                       Biogas_End_Use  \\\n",
       "7                         Electricity   \n",
       "8                         Electricity   \n",
       "9         Cogeneration; Refrigeration   \n",
       "10                   Electricity; CNG   \n",
       "27                        Electricity   \n",
       "..                                ...   \n",
       "489                      Cogeneration   \n",
       "490         Electricity; Cogeneration   \n",
       "491                      Cogeneration   \n",
       "492                      Cogeneration   \n",
       "504  Electricity; Boiler/Furnace fuel   \n",
       "\n",
       "        System Designer(s)_Developer(s) and Affiliates  \\\n",
       "7    4Creeks [System Designer, System Design Engine...   \n",
       "8    California Bioenergy LLC [Project Developer, S...   \n",
       "9    California Bioenergy LLC [Project Developer]; ...   \n",
       "10        California Bioenergy LLC [Project Developer]   \n",
       "27   Environmental Fabrics, Inc. [Biogas Membrane S...   \n",
       "..                                                 ...   \n",
       "489  Avatar Energy [Project Developer, System Desig...   \n",
       "490  Hadley and Bennett [System Designer, System De...   \n",
       "491  UEM Group [System Designer, System Design Engi...   \n",
       "492  Avatar Energy [Project Developer, System Desig...   \n",
       "504  AGES [Project Developer, System Designer, Syst...   \n",
       "\n",
       "                    Receiving Utility  Total Emission Reductions (MTCO2e/yr)   \\\n",
       "7                                PG&E                                 107,483   \n",
       "8                                PG&E                                  14,867   \n",
       "9                                   0                                  82,427   \n",
       "10   Pacific Gas and Electric Company                                  60,350   \n",
       "27   Pacific Gas and Electric Company                                  19,462   \n",
       "..                                ...                                     ...   \n",
       "489              Green Mountain Power                                       0   \n",
       "490                                 0                                       0   \n",
       "491      Vermont Electric Cooperative                                       0   \n",
       "492                    Stowe Electric                                       0   \n",
       "504                                 0                                       0   \n",
       "\n",
       "    Awarded USDA Funding? Biogas_ft3/cow  \n",
       "7                       0      38.709677  \n",
       "8                       0      29.411765  \n",
       "9                       0      27.835052  \n",
       "10                      0      51.428571  \n",
       "27                      Y      27.746032  \n",
       "..                    ...            ...  \n",
       "489                     0      31.250000  \n",
       "490                     0      98.684211  \n",
       "491                     0      95.571429  \n",
       "492                     0      41.666667  \n",
       "504                     0     111.111111  \n",
       "\n",
       "[67 rows x 22 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data analysis for dairy- all digester types\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Dairy farms host the majority of manure-based anaerobic digesters, therefore they have the most/best data for anaerobic digesters. The majority of analysis completed is for dairy anaerobic digesters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Biogas_ft3/cow', ylabel='Count'>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data = df2['Biogas_ft3/cow'], x=df2['Biogas_ft3/cow'], bins = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "73.46391320833166"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2['Biogas_ft3/cow'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "ci95_df2 = hist_filter_ci(df2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "69.73673816020273"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci95_df2['Biogas_ft3/cow'].mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "NOTE: Any \"efficiency\" calculation, outputs a value with units %"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "68.8173385176075"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "efficiency(ci95_df2['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "ci68_df2  = hist_filter_ci_68(df2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "65.78025898722731"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "efficiency(ci68_df2['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "dairy_biogas = pd.DataFrame(df2,columns=['Dairy', \"Biogas_gen_ft3_day\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A regression analysis and a 95% confident regression analysis (using  \"[0.025 and 0.0975]\" OLS Regression Result output) were completed for each digester type. Later, I realized this analysis isn't necessarily helpful for the goal of this particular task. Nonetheless, I left all regressions in this notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.635\n",
      "Model:                            OLS   Adj. R-squared:                  0.629\n",
      "Method:                 Least Squares   F-statistic:                     112.9\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           7.61e-16\n",
      "Time:                        16:13:07   Log-Likelihood:                -905.81\n",
      "No. Observations:                  67   AIC:                             1816.\n",
      "Df Residuals:                      65   BIC:                             1820.\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept   3584.7491   2.89e+04      0.124      0.902   -5.42e+04    6.14e+04\n",
      "Dairy         76.7459      7.224     10.623      0.000      62.318      91.174\n",
      "==============================================================================\n",
      "Omnibus:                       21.108   Durbin-Watson:                   0.994\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):               66.645\n",
      "Skew:                           0.791   Prob(JB):                     3.37e-15\n",
      "Kurtosis:                       7.623   Cond. No.                     5.20e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 5.2e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    }
   ],
   "source": [
    "dairy_biogas2 = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=dairy_biogas).fit()\n",
    "\n",
    "print(dairy_biogas2.summary())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Dairy', ylabel='Biogas_gen_ft3_day'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_biogas, ci =95)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<function matplotlib.pyplot.show(close=None, block=None)>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_biogas, ci = 95)\n",
    "\n",
    "Y_upper_dairy_biogas = dairy_biogas['Dairy']*91.174+.000614\n",
    "Y_lower_dairy_biogas = dairy_biogas['Dairy']*62.318-.000542\n",
    "\n",
    "plt.scatter(dairy_biogas['Dairy'], dairy_biogas['Biogas_gen_ft3_day'])\n",
    "plt.scatter(dairy_biogas['Dairy'], Y_upper_dairy_biogas, color = 'red')\n",
    "plt.scatter(dairy_biogas['Dairy'], Y_lower_dairy_biogas, color = 'red')\n",
    "\n",
    "plt.show"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "def filter_confidence_interval(data):\n",
    "    Y_upper = data['Dairy']*91.174+.000614\n",
    "    Y_lower = data['Dairy']*62.318-.000542\n",
    "    filtered_data = data[(data['Biogas_gen_ft3_day'] >= Y_lower) & (data['Biogas_gen_ft3_day'] <= Y_upper)]\n",
    "    return filtered_data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "ci95dairy_biogas = filter_confidence_interval(dairy_biogas)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x1b864e16f40>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(ci95dairy_biogas['Dairy'], ci95dairy_biogas['Biogas_gen_ft3_day'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.972\n",
      "Model:                            OLS   Adj. R-squared:                  0.970\n",
      "Method:                 Least Squares   F-statistic:                     518.7\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           4.77e-13\n",
      "Time:                        16:13:08   Log-Likelihood:                -199.71\n",
      "No. Observations:                  17   AIC:                             403.4\n",
      "Df Residuals:                      15   BIC:                             405.1\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept  -3956.9715   1.04e+04     -0.381      0.708   -2.61e+04    1.82e+04\n",
      "Dairy         77.0919      3.385     22.775      0.000      69.877      84.307\n",
      "==============================================================================\n",
      "Omnibus:                       12.189   Durbin-Watson:                   2.069\n",
      "Prob(Omnibus):                  0.002   Jarque-Bera (JB):               15.570\n",
      "Skew:                           0.786   Prob(JB):                     0.000416\n",
      "Kurtosis:                       7.417   Cond. No.                     4.03e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 4.03e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\scipy\\stats\\stats.py:1541: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=17\n",
      "  warnings.warn(\"kurtosistest only valid for n>=20 ... continuing \"\n"
     ]
    }
   ],
   "source": [
    "dairy_biogas3 = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=ci95dairy_biogas).fit()\n",
    "\n",
    "print(dairy_biogas3.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Plug flow digester type analysis for dairy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Creating dairy plug flow biogas generation\n",
    "\n",
    "df3 = df.rename(columns={\"Animal/Farm Type(s)\" : \"Animal\", \"Co-Digestion\" : \"Codigestion\", \"Biogas End Use(s)\" : \"Biogas_End_Use\", \" Biogas Generation Estimate (cu_ft/day) \" : \"Biogas_gen_ft3_day\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Covered Lagoon', 'Mixed Plug Flow', 'Unknown or Unspecified',\n",
       "       'Complete Mix', 'Horizontal Plug Flow', 0,\n",
       "       'Fixed Film/Attached Media',\n",
       "       'Primary digester tank with secondary covered lagoon',\n",
       "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor',\n",
       "       'Vertical Plug Flow', 'Complete Mix Mini Digester',\n",
       "       'Plug Flow - Unspecified', 'Dry Digester', 'Modular Plug Flow',\n",
       "       'Microdigester'], dtype=object)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3['Digester Type'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "df3.drop(df3[(df3['Animal'] != 'Dairy')].index, inplace = True)\n",
    "df3.drop(df3[(df3['Codigestion'] != 0)].index, inplace = True)\n",
    "df3.drop(df3[(df3['Biogas_gen_ft3_day'] == 0)].index, inplace = True)\n",
    "df3['Biogas_ft3/cow'] = df3['Biogas_gen_ft3_day'] / df3['Dairy']\n",
    "\n",
    "#df3.drop(df3[(df3['Biogas_End_Use'] == 0)].index, inplace = True)\n",
    "\n",
    "#selecting for 'Vertical Plug Flow', 'Horizontal Plug Flow', and 'Plug Flow - Unspecified', 'Modular Plug Flow', 'Mixed Plug FLow'\n",
    "\n",
    "notwant = ['Covered Lagoon', 'Unknown or Unspecified',\n",
    "       'Complete Mix', 0,\n",
    "       'Fixed Film/Attached Media',\n",
    "       'Primary digester tank with secondary covered lagoon',\n",
    "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor', 'Complete Mix Mini Digester', 'Dry Digester', \n",
    "       'Microdigester']\n",
    "\n",
    "df3 = df3[~df3['Digester Type'].isin(notwant)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Biogas_ft3/cow', ylabel='Count'>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data = df3['Biogas_ft3/cow'], bins = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "ci95_df3 = hist_filter_ci(df3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "74.73078118792917"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci95_df3['Biogas_ft3/cow'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "73.74554076333106"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "efficiency(ci95_df3['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "71.27049895301684"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci68_df3  = hist_filter_ci_68(df3)\n",
    "efficiency(ci68_df3['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(32, 22)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "dairy_plugflow = pd.DataFrame(df3,columns=['Dairy', \"Biogas_gen_ft3_day\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.779\n",
      "Model:                            OLS   Adj. R-squared:                  0.771\n",
      "Method:                 Least Squares   F-statistic:                     105.5\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           2.44e-11\n",
      "Time:                        16:13:09   Log-Likelihood:                -416.89\n",
      "No. Observations:                  32   AIC:                             837.8\n",
      "Df Residuals:                      30   BIC:                             840.7\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept   -1.38e+04   2.56e+04     -0.538      0.594   -6.61e+04    3.85e+04\n",
      "Dairy         98.8009      9.619     10.271      0.000      79.156     118.445\n",
      "==============================================================================\n",
      "Omnibus:                       43.234   Durbin-Watson:                   1.930\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):              225.237\n",
      "Skew:                           2.648   Prob(JB):                     1.23e-49\n",
      "Kurtosis:                      14.869   Cond. No.                     3.40e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 3.4e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYMAAAERCAYAAACZystaAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA6UElEQVR4nO3deXyc5Xno/d81i1Zblrwv0gibGMxqY2RbZCEOCQkQEkICwcZ2kjY9kKbpSdvTNmn7lrwvOZ/3TU572tAsBZqSNLbBrAGHsIQTQkwS5IXFYBsbjEGLLVvWYm2j0WzX+8fzaJCFJGvsGc12fT8ff6S5n2dmrgfsuea57+u+b1FVjDHGFDZPpgMwxhiTeZYMjDHGWDIwxhhjycAYYwyWDIwxxmDJwBhjDDmcDETkHhFpE5E9Ezz/8yKyT0T2isi96Y7PGGNyieTqPAMRuRzoA36mqhee4tzFwAPAFaraJSKzVbVtMuI0xphckLN3Bqq6Degc3iYiZ4vIUyLyoog8LyJL3EP/Dfihqna5z7VEYIwxw+RsMhjD3cCfq+qlwF8DP3LbzwHOEZHfi0iDiFyVsQiNMSYL+TIdQKqIyBTg/cCDIjLUXOz+9AGLgdVANfC8iFyoqicmOUxjjMlKeZMMcO5yTqjqslGOtQANqhoB3haRAzjJYeckxmeMMVkrb7qJVLUH54P+RgBxLHUPPwp8xG2fidNtdCgTcRpjTDbK2WQgIvcBLwDnikiLiHwZWAd8WUR2A3uB69zTnwY6RGQf8Bvgb1S1IxNxG2NMNkpraamI3ANcC7SNVf4pIquB7wF+oF1VP5y2gIwxxowq3clg3LkAIlIJ/AG4SlWbrP7fGGMyI60DyKq6TUTOGueUm4FHVLXJPX9CiWDmzJl61lnjvawxxpiRXnzxxXZVnTXasUxXE50D+EXkOWAqcIeq/my0E0XkFuAWgEAgwK5duyYtSGOMyQci0jjWsUwPIPuAS4FPAp8A/lFEzhntRFW9W1XrVLVu1qxRE5sxxpjTlOk7gxacQeN+oF9EtgFLgTcyG5YxxhSWTN8ZPAZ8SER8IlIGrAJez3BMxhhTcNJ6Z+DOBVgNzBSRFuBbOCWkqOqdqvq6iDwFvArEgR+r6oSWpDbGGJM66a4mWjuBc/4J+Kd0xmGMMWZ8me4mMsYYkwUsGRhjjLFkYIwxxpKBMcbkhFhc6Q1F0vb6lgyMMSbL9YYitHQFCYZjaXuPTE86M8YYM4ZILE573yADaUwCQywZGGNMFuoORugMhknnytLDWTIwxpgsMhiNcbx3kHA0Pqnva8nAGGOygKrS2R+meyB9g8TjsWRgjDEZNhCO0d43SCQ2uXcDw1kyMMaYDInFlY7+QfpC0UyHYsnAGGMyoTcUobM/TCw+OQPEp2LJwBhjJlEkFqejL0wwnPm7geEsGRhjzCTpDkboCoaJT1K5aDIsGRhjTJoNRmO094UZjKR/8tjpsmRgjDFpMlQu2hOKTtrksdOV1rWJROQeEWkTkXF3LxORFSISE5Eb0hmPMcZMloFwjJauAboHIlmfCCD9C9X9FLhqvBNExAt8F3g6zbEYY0zaxeJKW2+I1u6BjM4bSFZak4GqbgM6T3HanwMPA23pjMUYY9KtbzBKS1cwK+YNJCujYwYisgC4HrgCWHGKc28BbgEIBALpD84YYyYoW8tFk5Hp/Qy+B3xDVU85xK6qd6tqnarWzZo1K/2RGWPMBHQHIxzuGsjpRACZryaqA7aICMBM4BoRiarqoxmNyhhjTiEXykWTkdFkoKoLh34XkZ8Cj1siMMZkM1WlKxjJmSqhiUprMhCR+4DVwEwRaQG+BfgBVPXOdL63McakWjasLpouaU0Gqro2iXO/lMZQjDHmtGXT6qLpkukxA2OMyWp9g1E6+gazZnXRdLFkYIwxo4jG4rTneLloMiwZGGPMCNm8umi6WDIwxhhXvpWLJsOSgTGm4OVruWgyLBkYYwpaPpeLJsOSgTGmIMXizl4DvaFIpkPJCpYMjDEFp1DKRZNhycAYUzAKrVw0GZYMjDEFoXsgQld/YZWLJsOSgTEmr4WjcY73DRZkuWgyLBkYY/KSqnIiGOFEAZeLJsOSgTEm74QiTrloOFrY5aLJsGRgjMkbqk65aPeAlYsmy5KBMSYv2OSxM2PJwBiT0wphr4HJYMnAGJOzbPJY6qR728t7gGuBNlW9cJTj64BvuA/7gD9V1d3pjMkYk/uisTgd/WH6BwvnbmDHoU4efLGFY70haqrKuPXyRaxeMjtlr+9J2SuN7qfAVeMcfxv4sKpeDHwbuDvN8RhjclxPKEJL10DBJYI7nn2T9r4QlaV+2npD3LZ1L8/tb0vZe6Q1GajqNqBznON/UNUu92EDUJ3OeIwxuSsSi9PaPUB772DBzSLesrMZn0coLfIhIpQV+fB7hbu2HUrZe2TTmMGXgSfHOigitwC3AAQCgcmKyRiTBbqDETqD4YKdPNbaM0BFyckf16V+Ly1dwZS9R7q7iSZERD6Ckwy+MdY5qnq3qtapat2sWbMmLzhjTMYMRmMcPjFAR/9gwSYCgHkVpYQicVQ18d9hIBKjuqosZe+R8WQgIhcDPwauU9WOTMdjjMm8ocljR06EbE0h4Ka6avoGo7zTEeRYT4hgOEokptx6+aKUvUdGu4lEJAA8AmxQ1TcyGYsxJjuEIjGO99rkMXCS4guHOtjY0ERX0JlV3d4fZuHMKXx19dkprSZKd2npfcBqYKaItADfAvwAqnoncBswA/iRiABEVbUunTEZY7JTPK50BsP02FISxFX53ZvtbGpo4uDxvkT7pbVV/PXHz6V+0XTcz8yUSWsyUNW1pzj+J8CfpDMGY0z2C4ajtPeGicYL+24gFld++8ZxNm9v4u32/kT7irOq2FBfy6pFM5hTUZKW986maiJjTIGxpSQcsbjy69ePsXl7E81dA4n2yxbNYH19gPPmVaQ9BksGxpiMsKUknLkTv9p7jHt3NNHaHUq0f2jxTNavCrB4ztRJi8WSgTFmUtk+xM7ua0/uaeW+Hc209Q4CIMDqc2exvr6WhTPLJz0mSwbGmElT6PsQhyIxfvlaK1t2NtPRFwbAI/DR8+awbmWAwIzUzRtI1oSTgYhMV9Uxl5YwxpixhKNx2vsGCRXonIGBcIytu4/wwK7mRImo1yN84vw5rF0VYEFlaYYjTO7OYLuIvAL8BHhSC3k6oDFmQlTVuRsIFuY+xP2DUR595TAP7mqhxx0k93uFqy+cx5qVNcxNU2XQ6UgmGZwDfAz4Y+D7InI/8FObLGaMGU0h70PcG4rw8EuHeeSlw/S5q6sW+Txce/E8bqqrYdbU4gxH+F4TTgbuncAzwDPuWkKbgK+KyG7gm6r6QppiNMbkkELeh7g7GOGhl1r4+cuHCYadLrESv4dPL53P5+tqmF5elOEIx5bMmMEMYD2wATgG/DmwFVgGPAgsTEN8xpgcUqj7EHf2h3lgVzNbdx8hFHGuvazIy/WXLOCG5dVMK/NnOMJTS6ab6AVgI/AZVW0Z1r5LRO5MbVjGmFwSjysd/WF6Q4V1N3C8d5D7dzbz+Gutie6wKcU+Prd8AZ9dvoCpJdmfBIYkkwzOHWvQWFW/m6J4jDE5pn8wSkdfYS0lcbQnxJYdzTy5p5VIzPlYrCjxcWNdNZ9ZtoDy4tyr2k8m4pki8rfABUBiCFxVr0h5VMaYrBeLKx19g4kB0kJw+MQA9+1o4um9xxIzp6vK/Hy+roZPL51PaZE3wxGevmSSwWbgfpwN7r8CfBE4no6gjDHZrTcUobM/XDBLSTR1Btm8vYlfv36MoUueMaWItStq+ORF8yj2524SGJJMMpihqv8pIl9X1d8CvxWR36YrMGNM9onEnMljA+HCmDz2dns/mxoaee7AcYbS3uypxaxdGeDqC+dS5Mv4/mApk0wyGBoZahWRTwJHsA3sjSkYhbQP8ZvHetm0vYnn32xPtM2bVsK6VQGuPH8Ofm/+JIEhySSD/yki04D/AXwfqAD+Mi1RGWOyxmA0RntfuCC2n9x/tIeNLzTxwqF3d+Ctripl/aoAHz1vDl5PajeUySbJTDp73P21G/jIRJ4jIvfgjDG0qeqFoxwX4A7gGiAIfElVX5poTMaY9FFVTgQjnBjI/6Uk9hzuZmNDIzvf6Uq0nTWjjHWrall97qy8TgJDTpkMROT7wJh/E1T1v4/z9J8CPwB+Nsbxq4HF7p9VwL+7P40xGVQI+xCrKrtbuvnZC4280nwi0X72rHI21NfywcUz8aR4a8lsNpE7g13uzw8A5+NUFAHcCLw43hNVdZuInDXOKdcBP3PnLzSISKWIzFPV1gnEZYxJsULYh1hV2dXYxaaGRl473JNoP3fuVDbUB7hs0YyU7y+cC06ZDFT1vwBE5EvAR1Q14j6+E/jVGb7/AqB52OMWt+09yUBEbgFuAQgEAmf4tsaYkfJ9KQlVpeFQJ5u2N/J6a2+i/cL5Fayvr2XFWVUFmQSGJDOAPB+YCgztaTDFbTsTo/2XH2uW893A3QB1dXX53YFpzCTK932I46r8/mAHmxoaebOtL9G+rGYaG+prWVZTWdBJYEgyyeA7wMsi8hv38YeB//sM378FqBn2uBqnZNUYMwnyeR/iWFx5/s3jbGxo4u32/kR7XW0VG+pruah6Wgajyz7JVBP9RESe5N0B3m+q6tGh4yJygaruTfL9twJfE5Et7ut223iBMekXjcXp6A/Tn4dLScTiyrP729i8vYmmzmCivX7RdDbU13LevIoMRpe9klpNyf3wf2yMwxuB5cMbROQ+YDXOukYtwLcAv/tadwJP4JSVHsQpLf2jZOIxxiSvJxShsy//9iGOxuI8s+8Ym3c0ceREKNH+wffNZH19gHPmTM1gdNkvlUvrvafTTVXXjvcEt4roz1IYgzFmDPm6D3E4GufpvUe5b0czR3ucJCDA6nNnsW5VgEWzpmQ2wByRymSQX18zjMkT+boP8WAkxi9fa2XLzmba+8IAeASuWDKbdasC1M4oz3CEuSX3Ft02xkzYYNSZPJZP+xAPRGL8YvcR7t/ZTFfQmQ/h9QgfP38ON68MsKCqNMMR5qZUJoNwCl/LGHMGVJWuYITuPFpKon8wymOvHOHBF1sS+yv7PMLVF85l7coAc6eVnOIVzHgmlAxExAOgqnERKQIuBN5R1aE5B6hqfXpCNMYkI9+WkugLRXnk5RYefukwve5cCL9X+ORF81i7MsCsqcUZjjA/TGRtos8AdwFxEfkK8PdAP3COiPypqv4ivSEaYyYi3/Yh7h6I8NCLLTz68mH63f0TSnwePrV0Pp+vq2bGFEsCqTSRO4NvAUuBUmA3sEJVD4hILfAwYMnAmAzrG4zSmSf7EHcFwzy4q4VHXzlMKOJcT1mRl88sm88Nl1ZTWVaU4Qjz04S6iYYml4lIk6oecNsah7qPjDGZkU+Tx9r7Brl/ZzOPv9rKoDvgXV7s5XOXVPPZ5QuoKPVnOML8NuExA1WNA388rM0LWIo2JkO6ByJ09ef+5LG2nhD37WzmiddaicSca6ko8XHDpdV85pIFTCm2osfJMJH/yrfgfOiHVHXHsPYanPWKjDGTKF92HmvtHuDe7c08vfcoUXdtpKoyPzdeWs2nl82nrMiSwGSayBLWOwFE5Ouqesew9ndE5Lp0BmeMeVe+lIu2dAXZvL2JZ/YdY2h9vBnlRdy0ooZrL55Hid+b2QALVDKp94s4W1QO96VR2owxKZYPew2809HP5oYmfnOgLZEEZk8tZs2KGq65aB5FPhuCzKSJlJauBW4GFonI1mGHpgIdoz/LGJMK+bDXwFttfWzc3sjzb7Qn1qyZN62EtSsDfOKCOfi9lgSywUTuDBpwdh6bCfzvYe29wKvpCMoYk/t7DRw42svGhkb+8Na73xmrq0pZtyrAR5fMxmdJIKtMJBk8pKqXikhQVX+b9oiMKXCRWJyOvjDBcG7eDew53M2mhkZ2vNOVaKudXsb6+gCrz52N12O7imWjiSQDj4h8C2fG8V+NPKiq/5L6sIwpTN3BCF3B3CwX3d18gp81NPJy04lE29mzyllfX8uHFs/EY1tLZrWJJIM1wGfcc213CGPSIFfLRVWVFxu72NjQxGuHuxPt586Zyvr6AO8/e4btL5wjJlJaegD4roi8qqpPjnWeiHxRVf9rlParcCqOvMCPVfU7I45PAzYBATeef1bVnyR3GcbkplwtF1VVtr/dyaaGRva19ibaz59XwYbLAqw8a7olgRyTzB7IYyYC19eBk5KBO0v5h8CVQAuwU0S2quq+Yaf9GbBPVT8lIrOAAyKyWVVtSWyT13KxXDSuyh8OdrBpeyNvHOtLtF9cPY0N9bUsD1RaEshRad32ElgJHFTVQwDuxvfXAcOTgQJTxfkbNAXoBHJz5MyYCcjFctFYXHn+zeNsamjiUHt/ov3SQCXrL6tlaXVl5oIzKZHubS8XAM3DHrcAq0ac8wNgK3AEZ0ziJncdJGPyTm8oQmd/OGfKRWNx5TcH2tjc0ERjZzDRvmrhdDbU13L+/IoMRmdSKd13BqO1jfxX8AngFeAK4GzgGRF5XlV7TnohkVtw1kkiEAiccbDGTKZcKxeNxuI883ob925v4vCJgUT7B943gw31tZwzx2pJ8k0qk8HvR2lrwVnQbkg1zh3AcH8EfEed0bODIvI2sAQYvigeqno3cDdAXV1dbnytMganXLQzGM6JAeJwNM6v9h3l3u3NHO0JAc43ug+fM4t19QHOnjUlswGatJlwMhCRYuBzwFnDn6eqt7s/vzbK03YCi0VkIXAYp0z15hHnNAEfBZ4XkTnAucChiV+CMdkplzajD0fjPPFaK/ftaOZ43yAAHoErlsxm3aoAtTPKMxyhSbdk7gweA7qBF4HBiTxBVaMi8jXgaZzS0ntUda+7fSaqeifwbeCnIvIazpeQb6hqexJxGZNV4nGlKxhObNqezUKRGL94tZUHdjbT0e8U8HkErjx/DutWBaiuKstwhGayJJMMqlX1qmTfQFWfAJ4Y0XbnsN+PAB9P9nWNyUbBcJSOvnDWl4sGw1G2vnKEB3a1cMJNWj6PcNWFc1m7soZ500ozHKGZbMkkgz+IyEWq+lraojEmR8XiSkffIH1Zvv1kXyjKz18+zMMvtdDjlrb6vcInL5rHmhU1zK4oyXCEJlOSSQYfBL7kDvAO4nTpqKpenJbIjMkRuVAu2j0Q4eGXWvj5S4fpDztLXhT7PHxq6Tw+X1fDzCnFGY7QZFoyyeDqtEVhTA6KxOK09w0yEM7e9YS6gmEe3NXCY68cYcBd96jU7+W6ZfO5sa6aqjLbxtw4klmOolFEPggsVtWfuEtHWJ2ZKTiq6mxGH8ze9YQ6+ga5f1czv9jdyqBbzVRe5OX65Qv43PJqppX6MxyhyTbJlJZ+C6jDKf38CeDHWWDuA+kJzYzluf1t3LXtEM1dQWqqyrj18kWsXjI702EVhFDEWU8oW8tF23pCbNnZzC9fayUScxLV1BIfNyyv5vpLFjClxDaZN6NL5m/G9cAlwEvgVAGJiE1DnGTP7W/jtq178XuFylI/bb0hbtu6l9vBEkIaxeNKZzBMT5aWi7Z2D3Dfjmae2nOUqDt2Ma3Uz42XVnPdsvmUF1sSMONL5m9IWFVVRBRARGwWSgbcte0Qfq9QVuT8rysr8hEMR7lr2yFLBmkSDEdp7w0TjWff3cDhrgE2b2/iV/uOJjaZn15exE111Vy7dD6lfm9mAzQ5I5lk8ICI3AVUish/A/4Y+I/0hGXG0twVpHJEf2+p30tLV3CMZ5jTFY3F6ewPZ2W5aGNHP5u3N/Hs/rZEEpg1pZg1K2u45sK5FFsSMElKZgD5n0XkSqAHZ9zgNlV9Jm2RmVHVVJXR1htK3BkADERiNlM0xXpCETr7sm/7yUPH+9jU0MRv3zieWPFxbkUJN6+q4ePnz6XIZ5vMm9OTVEei++FvCSCDbr18Ebdt3UswHKXU72UgEiMSU269fFGmQ8sL4ahTLhrKsu0n3zjWy8aGRn5/sCPRtqCylHWrAnzsvNn4vJYEzJlJppqol/cuP90N7AL+x9AGNia9Vi+Zze04YwctXUGqrZooJVSVE8EIJ7Js+8l9R3rY2NDI9rc7E22108tYVx/gI+fOxuuxXcVMaiRzZ/AvOMtP34sz+3gNMBc4ANwDrE51cGZ0q5fMtg//FApFnNVFs2k9od0tJ9j0QiMvNp1ItC2aWc76+gAfWjzLkoBJuWSSwVWqOnyXsrtFpEFVbxeRv091YMakW7aVi6oqLzedYGNDI7tbuhPti2dPYUN9Le9/3ww8tr9wwfGIUFrkpcTvpawofYUBySSDuIh8HnjIfXzDsGPZc19tzAT0Dzqri2ZDuaiqsuOdTja+0MS+1nc3+Dtv3lQ21NeyauF022S+gIgIxT4PZW4CKJmkyrBkksE64A7gRzgf/g3AehEpBUbb2MaYrBONxenoD9OfBeWiqsof3upgU0MTB471JtovWlDBhvpaLq2tsiRQIPxe58O/tMhLic+LJwPdgMmUlh4CPjXG4d+JyN+p6v+XmrCMSb3ugQhd/ZkvF42r8vyb7WxqaOSt4/2J9ksClXyhvpalNZWZC85MCp/HQ0mRh1K/l1K/NyuqwVI5R/1GwJKByTrZUi4aiyvPHTjOpu2NNHa8O0lw5VlVrK+v5cIF0zIYnUknjwgl7gd/aZE3K+eDpDIZjHpfIyJX4XQveYEfq+p3RjlnNfA9nMXv2lX1wymMyxSobCkXjcWV//P6MTZvb6KlayDRftmiGWy4LMCSuRUZi82kT7H74V9W5KXY58n6Lr9UJoP3/GsTES/wQ+BKoAXYKSJbVXXfsHMqccYhrlLVJhGxmklzxrKhXDQSi/Orvce4d0cTrd0hwPnG9KFzZrJ+VS3vm20rwOcTv9dDaZE30fWTiX7/M5HuO4OVwMGhCWkisgW4Dtg37JybgUdUtQlAVdtSGJMpMPG40tEfpjeUuXLRcDTOk3tauW9HM229g4Czyfzqc2ezblWAhTNtjcd84PUIpX4vJW4C8GdBv/+ZSGUyeHCUtgVA87DHLcCqEeecA/hF5DlgKnCHqv5s5AuJyC3ALQCBQCAV8Zo80zcYpTOD5aKhSIxfvtbKlp3NdPSFAScJXHn+HG5eGaBmuq0flctEhBK/M+g7mSWfkyWZ5Sj+F/A/gQHgKWAp8BequglAVf/f0Z42StvI7iQfcCnwUaAUeMGdzPbGSU9SvRu4G6Curs7mNZiEaCxOe1+YYDgz5aID4RiP7T7Cg7ua6Qo6dyQ+j/CJC+aydmUN8ytLMxKXOXNFPo/b7++jxJ/9/f5nIpk7g4+r6t+KyPU43/BvBH6Ds9vZWFqAmmGPq3GWtBh5Truq9gP9IrINJ9G8gTGnkMly0b7BKI++fJiHXmyhJ+QkIr9XuObCeaxZWcOcipJJj8mcGZ/H7fd3u34KadmPZJLB0CL61wD3qWrnBLLkTmCxiCwEDuOsZ3TziHMeA34gIj6gCKcb6V+TiMsUoMFojPa+MIMZKBftGYjwyEuHefjlFvoHnfcv8nm49uJ5rFlRw8wpxZMekzk9w5d6KPVnZ8nnZEkmGfxCRPbjdBN9VURmAaHxnqCqURH5GvA0TmnpPaq6V0S+4h6/U1VfF5GngFeBOE756Z7TuRiT/1SVrmCE7gyUi54IhnnwxRYeffkIA24SKvF7uG7pfG6sq2F6edGkxmOSN7TUw1C9fy6UfE4WSeYflIhUAT2qGhORMqBCVY+mLbox1NXV6a5duyb7bU2GDYSdzegnu1y0o2+QB3a18IvdRwhFnfcuL/LymUsWcMPyaqaV+U/xCiaThko+yzK41EO2EJEXVbVutGPJVhMtAK4UkeGdoe+p/DEmlWJxpaN/kL7Q5A4QH+8dZMvOZn75WithNwlMKfbxueUL+OzyBUwtsSSQjbweOanePxuWesgFyVQTfQtnz4LzgSeAq4HfYcnApFHfYJSOvkFi8cnrEjraE+K+HU08tecokZjzvtNK/dx4aTXXLZtPeXEqK7LNmRKRxAd/SZGHYl9+lXxOlmT+Vt+AU+Xzsqr+kYjMAX6cnrBMoYvE4nRMcrno4a4B7t3RxK/2HUskn6oyPzetqOFTF8+nNI1ryZvkDC314NT8W79/KiSTDAZUNS4iURGpANoA23jXpFx3MEJXcPLKRZs6gmza3siz+9sYugGZOaWINStq+ORF8yjOs8lFucjv9SQ2dykpsJLPyZJMMtjlriP0H8CLQB+wIx1BmcI02eWib7f3s6mhkecOHE/MhJxTUczalQGuumBuQZcZZlq+LfWQC5LZz+Cr7q93uqWgFar6anrCMoVEVensD9MTik5Kueibx3rZ2NDE7w62J9rmV5awbmWAK8+fYwOOGZDvSz3kgmQGkJeP0nY20Kiqmd82yuSkySwXfb21h40NjTQc6ky0BaaXsW5VgCuWzLauh0lWSEs95IJkuol+BCzHmRwmwIXu7zNE5Cuq+qs0xGfy1GSWi+453M3PXmhkV2NXom3hzHI21Af40OJZlgQmyVC/fyEu9ZALkkkG7wBfVtW9ACJyPvA3wLeBRwBLBmZCekMROvvDaS0XVVVebj7BpoZGXmnuTrS/b/YUNtTX8oH3zcBj30TTaviHf4nPY91vWS6ZZLBkKBEAqOo+EblEVQ/Z7Z2ZiEjM2X5yIJy+AWJVZec7XWxsaGTvkZ5E+5K5U/nCZbWsWjjduiPSpMjnSfT32zf/3JNMMjggIv8ObHEf3wS8ISLFQOZ2EjE5oTsYoTMYTtsAsarywqEONjY0ceBob6L9ogUVrK+vpa62ypJAig31+Q8lAPvwz23JJIMvAV8F/gJnzOB3wF/jJIKPpDowkx8Go872k0PLOaRaXJXfvdnOpoYmDh7vS7Qvq6lkQ32AZTWVlgRSpNjvdPeU2ho/eSmZ0tIBEfk+ztiAAgdUdeiOoG/sZ5pCFI8rXcEw3QPpuWmMxZXfvnGcTQ2NvNMRTLSvOKuKDfW1XLhgWlret1CIyLBv/h778C8AyZSWrgb+C2cgWYAaEfmiqm5LS2Qmaz23v427th2iuStITVUZt16+iNVLZieOB8NROvrCaSkXjcWVX79+jM3bm2juGki01y+azob6Ws6bV5Hy9ywEQ0s7l9gSDwUrmW6i/42z29kBABE5B7gPZ8tKUyCe29/GbVv34vcKlaV+2npD3LZ1L7cDHzpnFh19g/QNpr5cNBKL88w+Jwm0dr+7jcaHFs9k/aoAi+dMTfl75rPh6/qX2Ie/IcmdzoYSAYCqviEitoZvgblr2yH8XqGsyPmrU1bkIxiO8sPnDrJwVnnKy0XD0ThP7jnKfTuaaOsdBJzb0tXnzmLdqgCLZk1J6fvlM5vkZcaT7NpE/wlsdB+vw1mjaFwichVwB85OZz9W1e+Mcd4KoAG4SVUfSiIuM4mau4JUlr77HUBV8XmE5s5gShPBYCTGL19r5b6dzXT0hQHwCHzsvDncvCpAYHpZyt4rX/k8HkqK3k0AVu1jxpNMMvhT4M+A/47z5WwbzqzkMYmIF/ghcCXOxvc7RWSrqu4b5bzv4myPabJYTVUZbb0hyop8xOJKNB5nIBxjbkVpSl5/IBxj6+4jPLCrma6gM/js9QifOH8Oa1cFWFCZmvfJRx6RRJ9/aVFh7+drkpdMNdEg8C/un4laCRxU1UMAIrIFuA7YN+K8PwceBlYk8domA269fBH/+NgeIrEwxT4PoUicaFxZs6LmjF63fzDKo68c5qEXDycqkPxe4eoL57FmZQ1zK0pO8QqFaWhd/zLbz9ecoVMmAxF5QFU/LyKvAe/pB1DVi8d5+gKgedjjFmDViNdfAFwPXME4yUBEbgFuAQgEAqcK26SBqnJxTSV/9pH3sWVHM0d7BphbUcqaFTWsXDT9tF6zNxThkZcO8/BLhxMDz0U+D9dePI+b6mqYNbU4lZeQ82xdf5MuE7kz+Lr789rTeP3R/qaOTCjfA76hqrHxvtWo6t3A3QB1dXWTtweiASAUcSaPRWJxVi6czsqFp/fhP6Q7GOGhl1p49OXD9LvLU5T4PHx62Xw+X1fD9PKiVISd87weOWlxN1vX36TLKZOBqra6PxuH2kRkJtChp15boAUY3n9QDRwZcU4dsMVNBDOBa0QkqqqPnjJ6k3bxuNLRH6Y3lJrJY539YR7Y1czW3UcIRZx5CGVFXj6zbD43XlrDtLLCLlAbKvkc+uZv6/qbyTKRbqJ64DtAJ84KpRtxPrQ9IvIFVX1qnKfvBBaLyELgMLAGuHn4Caq6cNh7/RR43BJBdugfdCaPReNnPnnseO8g9+9q5vFXWxNLU0wp9vHZ5Qv43PIFTC0p3CTg9zof/rbMg8mkiXQT/QD4e2Aa8Cxwtao2iMgSnElnYyYDVY2KyNdwqoS8wD2quldEvuIev/NML8CkXjQWp7M/nJLJY0d7QmzZ0cyTe1qJxJwbyYoSHzfWVXPdsgVMKU6moC0/eD2S6PYp9XttaWeTFSbyL9E3tHGNiNyuqg0Aqrp/IpULqvoE8MSItlGTgKp+aQLxmDRRVXoGoinZjP7IiQHu3dHE03uPJeYfVJX5ubGuhuuWzqe0qHC6P4a2dCzz+ygp8lDsK5xrN7ljIslgeB/BwIhjNpCbJ0IRZ/vJM11dtKkzyL3bm/g/rx9jaA7ajClFrFlRwycvmlcwfeBFPg9lRT5b58fkjIkkg6Ui0oNTGVTq/o772Iq/c5yq0hWMcCIYPqPXebu9n00NjTx34HjiG8LsqcWsXVnD1RfOy/sJULalo8l1E6kmKoyvcgVoeLno6TrY1semhka2vdmeaJs3rYSbVwb4+AVz8rYU0rZ0NPmm8EbvTEo2o99/tIeNLzTxwqGORFt1VSnrVwX46Hlz8u6bsd/rburitw9/k58sGRSYnlCEzr7THyDec7ibTQ2N7HinK9F21owy1tfX8uFzZuVNErD9fE2hsWRQIMJRZzP6UOT0NqPf3XyCnzU08nLTiUTb2bPK2VBfywcXz8ST4wOktp+vKXSWDPKcqnIiGOHEQCTpzehVlRcbu9jY0MRrh7sT7efOncqG+gCXLZqRs1Uy9uFvzMksGeSx0x0gVlUaDnWyaXsjr7f2JtovmF/BhvpaVpxVlXNJwDZzN2Z8lgzyUCQWpysYTnqAOK7K7w92sKmhkTfb+hLtS6unseGyWi6pqcyZJDD0zd8+/I2ZGEsGeSQWV7qCYXpD0aS6hGJxZdsbx9m0vYm32/sT7ZfWVrGhPsDF1ZVpiDa1vB5JfPjbEg/GJM+SQZ7oDUXocKuEdhzqZMvOZlp7Bpg3zn4Dsbjy6/1tbG5opLnr3cnl9Yums35VLefPr5jMS0hasd9L2dC3/wKZ2WxMulgyyBHP7W/jrm2HaO4KUlNVxq2XL2L1ktlEYk6V0IC7J8COQ53c8eyb+DxCRYmPjv5B7nj2Tb7O4kRCiMbiPLPvGJt3NHHkRCjxHh943wwuqa7idwfb+fYv942bSDJhaE/foWUebNDXmNSxZJADntvfxm1b9+L3CpWlftp6Q/zjY3v45uASLlgw7aQ5A1t2NuNzu0wASv1eBiIxtuxsZlmgkqf2HuW+HU0c6xkEnDVFPnzOLNbVB+joDZ8ykUymoQXehrp/bIE3Y9LHkkEOuGvbIfxeoazI+d9V4vcSicX5j+ff5l9uWnrSua09A1SUnPy/tcgnvNXex/r/3E57n7MGkUfgiiWzWbcqQO2McgB++OzuMRPJZCWD4ds6lvpt4NeYyWLJIAc0dwWpLPWjqsTizp9in4ejPSMXkYV5FaV09A9S6vcSV+XEQITO/jBxhV6cgdYrz5vDulUBFlSVnvTc0RJJiX/090kVj0iiz7/U7837Be2MyVaWDHJATVUZR3sGKPJ6E1VCoUicuRWl7zl3zYoa/vXXb9ATitAXiuLuJ4PXI1xz0VzWrggwd9roi80OTyRDxnqfM2HLOxuTfdL+NUxErhKRAyJyUES+OcrxdSLyqvvnDyKydLTXKVTxuHJTXQ2hSJxgOIqiDERiROPKmhU1J53bF4qy/1gPPQNRugecRCDA+xfN4N4/WcVffuycMRMBOIkkGndef7z3SZbXI0wp9jFrajGB6WVUV5UxvbyI0iKvJQJjskRa7wxExAv8ELgSaAF2ishWVd037LS3gQ+rapeIXA3cDaxKZ1y5YiDszCC+uGYaX79iMVt2NnO0Z4C5I6p8ugciPPxSCz9/6TD9blVRic/Dp5bO56YVNUwvL5rQ+61cNJ2rjs7hgRdbGIjEKPV7+fyl1ac1XmBln8bklnR3E60EDqrqIQAR2QJcBySSgar+Ydj5DUB1mmPKevG40tEfpjcUSbStXDT9PR/KXcEwD+5q4bFXjjDgLkBX6vdy/SXzueHSairLJpYEhuw41MlT+44xvbyIEr+HUCTOU/uOce7cilMmBCv7NCa3pTsZLACahz1uYfxv/V8GnkxrRFluIusJtfcNcv/OZh5/tZVBd5vK8mIvn7ukms8uX0BFqf+03nu8stSRycDKPo3JL+lOBqN9PRx1nQQR+QhOMvjgGMdvAW4BCAQCqYova0xkw5ljPSG27GjmiT2tRNyR4YoSH5+7tJrrL1nAlOIz+995qmqiYrfixwZ+jck/6U4GLcDw0cdq4MjIk0TkYuDHwNWq2jHyOICq3o0znkBdXd3p7cyShVSVnlCUrv6xN5w5cmKA+3Y08/Teo0TdXeYrS/18vq6aTy+bn5h/cKZGVhOJCIPROIHp5dTOKLeuH2PyWLqTwU5gsYgsBA4Da4Cbh58gIgHgEWCDqr6R5niySjAcpaMvPGaXUHNnkHt3NPHMvmO4OYAZ5UXctKKGay+el/KB2ZtX1XDHrw8SicUpK/ISisZRha+uPtsSgTF5Lq3JQFWjIvI14GnAC9yjqntF5Cvu8TuB24AZwI/cboeoqtalM65MG4zG6OqPEAyP3iX0dns/m7c38dyBtkQSmDWlmLUra7jmonkpm5g1tNJniTvbd9GsKcyeWsJd2w7R0hWketgaSMaY/CbJ7n6VDerq6nTXrl2ZDiNpkVicrv4wfYOjJ4G32vrYuL2R599oTwyszJtWwtqVAT5xwRz8Z7gss0ckMdO3pMhjg77GFBgReXGsL9s2A/k0jLWC6FjnVFeWsnZlgItrKkfdZ+DA0V42NjTyh7feHS6Z6U7KGozGePb1NmZPKT7jev9inw36GmNGZ3cGSRq+guhQ6WUkptz+6QsSCWHoHJ/HWXohGHbO+foVJ6/+ufdINxsbmtjxdmeirXZGGZctnMFzb7S5i7Y59f7R+HufPxqvx1nrx+r9jTEj2Z1BCo1cQbSsyEcwHOWubYcSyeCubYfwesDv9RKPKyU+L6rv1uvvbjnBxhcaeanpROJ1z55Vzvr6Wj60eCZ//cCr+L2eCa8emljl0+r9jTGnyZJBkoZWEB2u1O+lpSsIOOMC73T0IyhHu0NEYnH8Xg+VpT4aO/v5i/tf4dWW7sRzz50zlfX1Ad5/9oxEF05jZz8D4SjRuOL3epheXkRZkTdR758L3/4n0pVmjMkelgySVFNVRltv6KTa/oFIjAWVpbT3DdIbilLq89DYGcQjggDhaJzWHmcfga6gkwjOn1fBFy6rZcVZVSf14+841EnfoLOHsccjRONKW88gVeV+AtPLWVBVmvXf/kfbjOe2rXu5HSwhGJOlbPH4JN16+SIiMXVWEFWlfzBCKBLn+ksW0DMQcQaIRVCFOEo0TmIZaYCl1dP45xsu5vtrl7Fy4fT3DOhu2dnMtKFZwHHnf5CidA9E+dpH3pf1iQBO7koTcX76vcJd2w5lOjRjzBjszuAURuvuuP3TF3Dnb9+iqTPI7IoS1tTVsGKh05cfV2dZiaFv9UOKfMKUIh//etOyUd9nqOunrS/EnIoSyov9tPcNEo7FKfZ58Yhy17ZD/F+P7cn6bpdTdaUZY7KPJYNxjLX38N9+4ly+87mLT1o+IhZXfnOgjc0NTXQPvDuPoKzIywx3CekZ5cUnvb7XI5QX+ygv8lFa5Hzjr51eTltviIpSf2LBueO9IbqCEdp6QznR7TJWV1p1VVkGozLGjMeSwTiGd3eoKkVeD+FonP/83TusWVHDlp3NHOkOUuzzEgzH6OgPJ55b7PNQWeqnotSXKA1ds6IGjzgJYErxuwlguFsvX8RtW/cSDEcTVURdwQjTy/3jVjBlk9GuIRJTbr18UaZDM8aMwZLBOJq7gkwr8RGNxYmpgjqreDZ29vO9X7/BYDRObyhKNO4kAQEuP2cW61cF6OgLn7QZzRcvq+XjF86l7BS7e61eMpvb4aQlIboHIu+5q8jmbpfRriGbu7WMMZYMxhSNxSn1eXizrY+4KoKzimc0piggAsOGBCgr8hKoKuPqC+byw9+8RWvPAPOnlfIP15zHVRfNS2opidVLZp/0wbn27oac63YZeQ3GmOxmyWCESCzOiWCEX79+jCMnBggPLwUathXD0HBBRYmP6WVF+H3Csd4Qdzz7Jn6vML2siO6BMP/0qzeYWuI/ow/GWy9fxN88tJvDXQNE43F8Hg9TS3z84yfPP+3XNMaY4SwZ4AwU/7tbHTSnooRLqqdx784mBqNjL9UhOBVCcyucDeYHIjEGwlGC4ShxhSKvh5lTihMllWf6LVndNxURkDF2CDLGmNNU0MlAVXlqz1G+/ct9eEWYUuzlcFc/r7acOKkLaCSvOHcGTpeREo7GCYajhGOKzyN43e6kI90DzJ9WcsZ9+3dtO8S0Uj/zppUm2rJ5ANkYk3sKctJZKBKjvW+Qps4gd/32EF5xFp0ThN5QlFOt3RdTiLs/GzuClBX5mD21hCKvsyqoiODxCB6EY72DZ9y339wVTKxTNCSbB5CNMbmn4O4MjvcO0huKsONQJ1t2NvPq4RMUeYXKsiL6BqOExukaGqnYK1RPL6M/HCMYjjKnopjW7kHiKCLOzOFojDMuqbS6fWNMuhVMMnhufxvffWo/bx13qoNUoarMT5FXGIwqR3sGk3o9v8e5Oxiq+Q9H40wr9TO/soTjvc7MYa8IZ88qP+OuHKvbN8akW9q7iUTkKhE5ICIHReSboxwXEfk39/irIrI81TE8t7+Nv3loN2+29aH67npB7f0RQlE9aTB2Iut/FnkFj3gocstFS/1eirxCJKZ4PcLCmeUEppcxu6KEb1y15IzjX71kNrd/+gJmTy2heyDC7KklJ+2fYIwxZyqtdwYi4gV+CFwJtAA7RWSrqu4bdtrVwGL3zyrg392fKXPXtkP0hqJ4PeKOB5zcFeQZNmegxO/sI9A3GE2Ulfo8EHX3rPe5lTyKMnPKu5VEi+dUcOvli9I20crq9o0x6ZTubqKVwEFVPQQgIluA64DhyeA64GfqbLnWICKVIjJPVVtTFURzV5BIzPk0j40YEvAILJpZTlNXEBQC08vxegSvR2jvC9E/GGNaqZ8pxc6SFO39YXpDUarK/EwtcbqIhrps7APbGJOr0p0MFgDNwx638N5v/aOdswA4KRmIyC3ALQCBQCCpIGqqyjjeExoxgcxR7PMwGI1TXuTDKxCNx/F7vQTDUfxeL/+25uIx9ze2pRaMMfki3clgtC74kZ/IEzkHVb0buBucPZCTCcKZwdtLR38Yj/uGcWfbAUp8HuZNK+Wrq88GJraejt0BGGPyTbqTQQtQM+xxNXDkNM45I6uXzOafbliaqCZSYGFVKX971RI+ceG895xrjDGFJt3JYCewWEQWAoeBNcDNI87ZCnzNHU9YBXSncrxgyNC3+a7+MMV+z0k1+8YYU+jS+omoqlER+RrwNOAF7lHVvSLyFff4ncATwDXAQSAI/FE6Y6pyN5oxxhjzrrR/PVbVJ3A+8Ie33TnsdwX+LN1xGGOMGVtBrk1kjDHmZJYMjDHGWDIwxhhjycAYYwyWDIwxxmDJwBhjDJYMjDHGYMnAGGMMIHqqDX+zkIgcBxqTeMpMoD1N4Uy2fLoWyK/rsWvJTvl0LXBm11OrqrNGO5CTySBZIrJLVesyHUcq5NO1QH5dj11Ldsqna4H0XY91ExljjLFkYIwxpnCSwd2ZDiCF8ulaIL+ux64lO+XTtUCarqcgxgyMMcaMr1DuDIwxxozDkoExxpj8TwYicpWIHBCRgyLyzUzHM5KI1IjIb0TkdRHZKyJfd9uni8gzIvKm+7Nq2HP+zr2eAyLyiWHtl4rIa+6xfxMRydA1eUXkZRF5PA+upVJEHhKR/e7/o8ty9XpE5C/dv2N7ROQ+ESnJpWsRkXtEpE1E9gxrS1n8IlIsIve77dtF5KxJvpZ/cv+evSoiPxeRykm9FlXN2z84W22+BSwCioDdwPmZjmtEjPOA5e7vU4E3gPOB/wV8023/JvBd9/fz3esoBha61+d1j+0ALgMEeBK4OkPX9FfAvcDj7uNcvpb/Av7E/b0IqMzF6wEWAG8Dpe7jB4Av5dK1AJcDy4E9w9pSFj/wVeBO9/c1wP2TfC0fB3zu79+d7GuZ9H9ck/wP4DLg6WGP/w74u0zHdYqYHwOuBA4A89y2ecCB0a4BZ3/py9xz9g9rXwvclYH4q4FfA1fwbjLI1WupwPkAlRHtOXc9OMmgGZiOs93t4+6HT05dC3DWiA/QlMU/dI77uw9nlq9M1rWMOHY9sHkyryXfu4mG/gEMaXHbspJ7K3cJsB2Yo6qtAO7P2e5pY13TAvf3ke2T7XvA3wLxYW25ei2LgOPAT9xurx+LSDk5eD2qehj4Z6AJaAW6VfVX5OC1jJDK+BPPUdUo0A3MSFvk4/tjnG/6J8XlSsu15HsyGK0vMytraUVkCvAw8Beq2jPeqaO06Tjtk0ZErgXaVPXFiT5llLasuBaXD+dW/t9V9RKgH6crYixZez1uX/p1ON0M84FyEVk/3lNGacuKa5mg04k/K65NRP4BiAKbh5pGOS3l15LvyaAFqBn2uBo4kqFYxiQifpxEsFlVH3Gbj4nIPPf4PKDNbR/rmlrc30e2T6YPAJ8WkXeALcAVIrKJ3LwW3DhaVHW7+/ghnOSQi9fzMeBtVT2uqhHgEeD95Oa1DJfK+BPPEREfMA3oTFvkoxCRLwLXAuvU7eNhkq4l35PBTmCxiCwUkSKcgZStGY7pJO7o/38Cr6vqvww7tBX4ovv7F3HGEoba17jVAguBxcAO9xa5V0Tq3df8wrDnTApV/TtVrVbVs3D+Wz+rqutz8VoAVPUo0Cwi57pNHwX2kZvX0wTUi0iZG8NHgdfJzWsZLpXxD3+tG3D+/k7anYGIXAV8A/i0qgaHHZqca5msgZ9M/QGuwanQeQv4h0zHM0p8H8S5fXsVeMX9cw1O/96vgTfdn9OHPecf3Os5wLBKDqAO2OMe+wFpHPyawHWt5t0B5Jy9FmAZsMv9//MoUJWr1wP8P8B+N46NONUpOXMtwH044x0RnG++X05l/EAJ8CBwEKdKZ9EkX8tBnH7+oc+BOyfzWmw5CmOMMXnfTWSMMWYCLBkYY4yxZGCMMcaSgTHGGCwZGGOMwZKBMeMSkZiIvCLOap+7ReSvRGTcfzciMl9EHpqsGI1JBSstNWYcItKnqlPc32fjrMb6e1X91mm8lk+ddWKMyTqWDIwZx/Bk4D5ehDOzfSZQizN5q9w9/DVV/YO74ODjqnqhiHwJ+CTOJKBy4DDwkKo+5r7eZpzlhbNqZrwpPL5MB2BMLlHVQ2430WycdXCuVNWQiCzGmVVaN8rTLgMuVtVOEfkw8JfAYyIyDWd9oC+O8hxjJpUlA2OSN7QipB/4gYgsA2LAOWOc/4yqdgKo6m9F5Idul9NngYet68hkA0sGxiTB7SaK4dwVfAs4BizFKcYIjfG0/hGPNwLrcBbz++P0RGpMciwZGDNBIjILuBP4gaqq283Toqpxd+lh7wRf6qc4i4cdVdW96YnWmORYMjBmfKUi8gpOl1AU51v90FLjPwIeFpEbgd/w3juAUanqMRF5HWcVVGOyglUTGTPJRKQMeA1YrqrdmY7HGLBJZ8ZMKhH5GM6eAt+3RGCyid0ZGGOMsTsDY4wxlgyMMcZgycAYYwyWDIwxxmDJwBhjDPD/A4e+mnD9Q7AeAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_plugflow, ci = 95)\n",
    "dairy_plugsum = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=dairy_plugflow).fit()\n",
    "print(dairy_plugsum.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<function matplotlib.pyplot.show(close=None, block=None)>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_plugflow, ci = 95)\n",
    "\n",
    "Y_upper_dairy_biogas_pf = dairy_plugflow['Dairy']*118.445+.000385\n",
    "Y_lower_dairy_biogas_pf = dairy_plugflow['Dairy']*79.156-.000661\n",
    "\n",
    "plt.scatter(dairy_plugflow['Dairy'], dairy_plugflow['Biogas_gen_ft3_day'])\n",
    "plt.scatter(dairy_plugflow['Dairy'], Y_upper_dairy_biogas_pf, color = 'red')\n",
    "plt.scatter(dairy_plugflow['Dairy'], Y_lower_dairy_biogas_pf, color = 'red')\n",
    "\n",
    "plt.show"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "def filter_confidence_interval_pf(data):\n",
    "    Y_upper = data['Dairy']*118.445+.000385\n",
    "    Y_lower = data['Dairy']*79.156-.000661\n",
    "    filtered_data = data[(data['Biogas_gen_ft3_day'] >= Y_lower) & (data['Biogas_gen_ft3_day'] <= Y_upper)]\n",
    "    return filtered_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "ci95dairy_biogas_pf = filter_confidence_interval_pf(dairy_plugflow)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x1b8650d0bb0>"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(ci95dairy_biogas_pf['Dairy'], ci95dairy_biogas_pf['Biogas_gen_ft3_day'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.997\n",
      "Model:                            OLS   Adj. R-squared:                  0.996\n",
      "Method:                 Least Squares   F-statistic:                     2254.\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           4.81e-10\n",
      "Time:                        16:13:09   Log-Likelihood:                -101.65\n",
      "No. Observations:                   9   AIC:                             207.3\n",
      "Df Residuals:                       7   BIC:                             207.7\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept  -7558.4626   8961.461     -0.843      0.427   -2.87e+04    1.36e+04\n",
      "Dairy        100.0387      2.107     47.475      0.000      95.056     105.021\n",
      "==============================================================================\n",
      "Omnibus:                       14.777   Durbin-Watson:                   3.114\n",
      "Prob(Omnibus):                  0.001   Jarque-Bera (JB):                6.502\n",
      "Skew:                          -1.746   Prob(JB):                       0.0387\n",
      "Kurtosis:                       5.268   Cond. No.                     5.19e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 5.19e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\scipy\\stats\\stats.py:1541: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=9\n",
      "  warnings.warn(\"kurtosistest only valid for n>=20 ... continuing \"\n"
     ]
    }
   ],
   "source": [
    "dairy_biogas4 = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=ci95dairy_biogas_pf).fit()\n",
    "\n",
    "print(dairy_biogas4.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Complete mix digester type analysis for dairy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4 = df.rename(columns={\"Animal/Farm Type(s)\" : \"Animal\", \"Co-Digestion\" : \"Codigestion\", \"Biogas End Use(s)\" : \"Biogas_End_Use\", \" Biogas Generation Estimate (cu_ft/day) \" : \"Biogas_gen_ft3_day\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Covered Lagoon', 'Mixed Plug Flow', 'Unknown or Unspecified',\n",
       "       'Complete Mix', 'Horizontal Plug Flow', 0,\n",
       "       'Fixed Film/Attached Media',\n",
       "       'Primary digester tank with secondary covered lagoon',\n",
       "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor',\n",
       "       'Vertical Plug Flow', 'Complete Mix Mini Digester',\n",
       "       'Plug Flow - Unspecified', 'Dry Digester', 'Modular Plug Flow',\n",
       "       'Microdigester'], dtype=object)"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4['Digester Type'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4.drop(df4[(df4['Animal'] != 'Dairy')].index, inplace = True)\n",
    "df4.drop(df4[(df4['Codigestion'] != 0)].index, inplace = True)\n",
    "df4.drop(df4[(df4['Biogas_gen_ft3_day'] == 0)].index, inplace = True)\n",
    "df4['Biogas_ft3/cow'] = df4['Biogas_gen_ft3_day'] / df4['Dairy']\n",
    "\n",
    "\n",
    "#df4.drop(df4[(df4['Biogas_End_Use'] == 0)].index, inplace = True)\n",
    "\n",
    "#selecting for 'Complete Mix', 'Complete Mix Mini Digester'\n",
    "\n",
    "notwant = ['Covered Lagoon', 'Mixed Plug Flow', 'Unknown or Unspecified', 'Horizontal Plug Flow', 0,\n",
    "       'Fixed Film/Attached Media',\n",
    "       'Primary digester tank with secondary covered lagoon',\n",
    "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor',\n",
    "       'Vertical Plug Flow','Plug Flow - Unspecified', 'Dry Digester', 'Modular Plug Flow',\n",
    "       'Microdigester']\n",
    "\n",
    "df4 = df4[~df4['Digester Type'].isin(notwant)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(14, 22)"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Biogas_ft3/cow', ylabel='Count'>"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data = df4['Biogas_ft3/cow'], bins = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "87.083266948478"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci95_df4 = hist_filter_ci(df4)\n",
    "\n",
    "ci95_df4['Biogas_ft3/cow'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "85.9351730367076"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "efficiency(ci95_df4['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "82.74672041445051"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci68_df4  = hist_filter_ci_68(df4)\n",
    "efficiency(ci68_df4['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.835\n",
      "Model:                            OLS   Adj. R-squared:                  0.822\n",
      "Method:                 Least Squares   F-statistic:                     60.92\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           4.84e-06\n",
      "Time:                        16:13:10   Log-Likelihood:                -188.22\n",
      "No. Observations:                  14   AIC:                             380.4\n",
      "Df Residuals:                      12   BIC:                             381.7\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept  -3.866e+04   6.77e+04     -0.571      0.579   -1.86e+05    1.09e+05\n",
      "Dairy        115.0487     14.740      7.805      0.000      82.933     147.165\n",
      "==============================================================================\n",
      "Omnibus:                        1.294   Durbin-Watson:                   2.405\n",
      "Prob(Omnibus):                  0.524   Jarque-Bera (JB):                0.180\n",
      "Skew:                          -0.227   Prob(JB):                        0.914\n",
      "Kurtosis:                       3.320   Cond. No.                     6.46e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 6.46e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n",
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\scipy\\stats\\stats.py:1541: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=14\n",
      "  warnings.warn(\"kurtosistest only valid for n>=20 ... continuing \"\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "dairy_cm = pd.DataFrame(df4,columns=['Dairy', \"Biogas_gen_ft3_day\"])\n",
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_cm, ci = 95)\n",
    "dairy_plugsum = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=dairy_cm).fit()\n",
    "print(dairy_plugsum.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<function matplotlib.pyplot.show(close=None, block=None)>"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_cm, ci = 95)\n",
    "\n",
    "Y_upper_dairy_biogas_cm = dairy_cm['Dairy']*147.165+.00001\n",
    "Y_lower_dairy_biogas_cm = dairy_cm['Dairy']*82.933 -.0000186\n",
    "\n",
    "plt.scatter(dairy_cm['Dairy'], dairy_cm['Biogas_gen_ft3_day'])\n",
    "plt.scatter(dairy_cm['Dairy'], Y_upper_dairy_biogas_cm, color = 'red')\n",
    "plt.scatter(dairy_cm['Dairy'], Y_lower_dairy_biogas_cm, color = 'red')\n",
    "\n",
    "plt.show"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.923\n",
      "Model:                            OLS   Adj. R-squared:                  0.908\n",
      "Method:                 Least Squares   F-statistic:                     59.92\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           0.000575\n",
      "Time:                        16:13:10   Log-Likelihood:                -91.312\n",
      "No. Observations:                   7   AIC:                             186.6\n",
      "Df Residuals:                       5   BIC:                             186.5\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept  -1.714e+04   7.15e+04     -0.240      0.820   -2.01e+05    1.67e+05\n",
      "Dairy        120.3140     15.542      7.741      0.001      80.361     160.267\n",
      "==============================================================================\n",
      "Omnibus:                          nan   Durbin-Watson:                   3.105\n",
      "Prob(Omnibus):                    nan   Jarque-Bera (JB):                0.132\n",
      "Skew:                          -0.295   Prob(JB):                        0.936\n",
      "Kurtosis:                       3.324   Cond. No.                     6.56e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 6.56e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\statsmodels\\stats\\stattools.py:74: ValueWarning: omni_normtest is not valid with less than 8 observations; 7 samples were given.\n",
      "  warn(\"omni_normtest is not valid with less than 8 observations; %i \"\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def filter_confidence_interval_cm(data):\n",
    "    Y_upper = data['Dairy']*147.165+.00001\n",
    "    Y_lower = data['Dairy']*82.933 -.0000186\n",
    "    filtered_data = data[(data['Biogas_gen_ft3_day'] >= Y_lower) & (data['Biogas_gen_ft3_day'] <= Y_upper)]\n",
    "    return filtered_data\n",
    "\n",
    "ci95dairy_biogas_cm = filter_confidence_interval_cm(dairy_cm)\n",
    "\n",
    "plt.scatter(ci95dairy_biogas_cm['Dairy'], ci95dairy_biogas_cm['Biogas_gen_ft3_day'])\n",
    "\n",
    "dairy_biogas5 = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=ci95dairy_biogas_cm).fit()\n",
    "\n",
    "print(dairy_biogas5.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Impermeable cover digester type analysis for dairy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "df5 = df.rename(columns={\"Animal/Farm Type(s)\" : \"Animal\", \"Co-Digestion\" : \"Codigestion\", \"Biogas End Use(s)\" : \"Biogas_End_Use\", \" Biogas Generation Estimate (cu_ft/day) \" : \"Biogas_gen_ft3_day\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Covered Lagoon', 'Mixed Plug Flow', 'Unknown or Unspecified',\n",
       "       'Complete Mix', 'Horizontal Plug Flow', 0,\n",
       "       'Fixed Film/Attached Media',\n",
       "       'Primary digester tank with secondary covered lagoon',\n",
       "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor',\n",
       "       'Vertical Plug Flow', 'Complete Mix Mini Digester',\n",
       "       'Plug Flow - Unspecified', 'Dry Digester', 'Modular Plug Flow',\n",
       "       'Microdigester'], dtype=object)"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5['Digester Type'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "df5.drop(df5[(df5['Animal'] != 'Dairy')].index, inplace = True)\n",
    "df5.drop(df5[(df5['Codigestion'] != 0)].index, inplace = True)\n",
    "df5.drop(df5[(df5['Biogas_gen_ft3_day'] == 0)].index, inplace = True)\n",
    "df5['Biogas_ft3/cow'] = df5['Biogas_gen_ft3_day'] / df5['Dairy']\n",
    "\n",
    "#df5.drop(df5[(df5['Biogas_End_Use'] == 0)].index, inplace = True)\n",
    "\n",
    "#selecting for 'Covered Lagoon'\n",
    "\n",
    "notwant = ['Mixed Plug Flow', 'Unknown or Unspecified',\n",
    "       'Complete Mix', 'Horizontal Plug Flow', 0,\n",
    "       'Fixed Film/Attached Media',\n",
    "       'Primary digester tank with secondary covered lagoon',\n",
    "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor',\n",
    "       'Vertical Plug Flow', 'Complete Mix Mini Digester',\n",
    "       'Plug Flow - Unspecified', 'Dry Digester', 'Modular Plug Flow',\n",
    "       'Microdigester']\n",
    "\n",
    "df5 = df5[~df5['Digester Type'].isin(notwant)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Biogas_ft3/cow', ylabel='Count'>"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data = df5['Biogas_ft3/cow'], bins = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "43.68490677908262"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci95_df5 = hist_filter_ci(df5)\n",
    "\n",
    "ci95_df5['Biogas_ft3/cow'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "43.108970927491335"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "efficiency(ci95_df5['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "41.088176141031695"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci68_df5  = hist_filter_ci_68(df5)\n",
    "efficiency(ci68_df5['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(15, 22)"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.896\n",
      "Model:                            OLS   Adj. R-squared:                  0.888\n",
      "Method:                 Least Squares   F-statistic:                     111.6\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           9.46e-08\n",
      "Time:                        16:13:10   Log-Likelihood:                -183.88\n",
      "No. Observations:                  15   AIC:                             371.8\n",
      "Df Residuals:                      13   BIC:                             373.2\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept   1.424e+04   1.92e+04      0.742      0.471   -2.72e+04    5.57e+04\n",
      "Dairy         36.8211      3.485     10.566      0.000      29.293      44.350\n",
      "==============================================================================\n",
      "Omnibus:                        0.535   Durbin-Watson:                   2.801\n",
      "Prob(Omnibus):                  0.765   Jarque-Bera (JB):                0.154\n",
      "Skew:                           0.240   Prob(JB):                        0.926\n",
      "Kurtosis:                       2.873   Cond. No.                     7.47e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 7.47e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n",
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\scipy\\stats\\stats.py:1541: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=15\n",
      "  warnings.warn(\"kurtosistest only valid for n>=20 ... continuing \"\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "dairy_ic = pd.DataFrame(df5,columns=['Dairy', \"Biogas_gen_ft3_day\"])\n",
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_ic, ci = 95)\n",
    "dairy_plugsum = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=dairy_ic).fit()\n",
    "print(dairy_plugsum.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Impermeable cover anaerobic digesters are not heated. I assumed biogas per dairy would vary significantly per location for impermeable cover digesters. That does not appear to be the case with the below preliminary analysis graph."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Dairy', ylabel='Biogas_gen_ft3_day'>"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot('Dairy', 'Biogas_gen_ft3_day', data = df5, hue = 'State')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<function matplotlib.pyplot.show(close=None, block=None)>"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_ic, ci = 95)\n",
    "\n",
    "Y_upper_dairy_biogas_ic = dairy_ic['Dairy']*44.35+.000557\n",
    "Y_lower_dairy_biogas_ic = dairy_ic['Dairy']*29.293 -.000272\n",
    "\n",
    "plt.scatter(dairy_ic['Dairy'], dairy_ic['Biogas_gen_ft3_day'])\n",
    "plt.scatter(dairy_ic['Dairy'], Y_upper_dairy_biogas_ic, color = 'red')\n",
    "plt.scatter(dairy_ic['Dairy'], Y_lower_dairy_biogas_ic, color = 'red')\n",
    "\n",
    "plt.show\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.998\n",
      "Model:                            OLS   Adj. R-squared:                  0.997\n",
      "Method:                 Least Squares   F-statistic:                     1122.\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           0.000890\n",
      "Time:                        16:13:11   Log-Likelihood:                -42.415\n",
      "No. Observations:                   4   AIC:                             88.83\n",
      "Df Residuals:                       2   BIC:                             87.60\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept  -1.519e+04   9556.189     -1.590      0.253   -5.63e+04    2.59e+04\n",
      "Dairy         39.4642      1.178     33.489      0.001      34.394      44.534\n",
      "==============================================================================\n",
      "Omnibus:                          nan   Durbin-Watson:                   2.389\n",
      "Prob(Omnibus):                    nan   Jarque-Bera (JB):                0.248\n",
      "Skew:                          -0.225   Prob(JB):                        0.883\n",
      "Kurtosis:                       1.867   Cond. No.                     1.12e+04\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 1.12e+04. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\statsmodels\\stats\\stattools.py:74: ValueWarning: omni_normtest is not valid with less than 8 observations; 4 samples were given.\n",
      "  warn(\"omni_normtest is not valid with less than 8 observations; %i \"\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def filter_confidence_interval_ic(data):\n",
    "    Y_upper = data['Dairy']*44.35+.000557\n",
    "    Y_lower = data['Dairy']*29.293 -.000272\n",
    "    filtered_data = data[(data['Biogas_gen_ft3_day'] >= Y_lower) & (data['Biogas_gen_ft3_day'] <= Y_upper)]\n",
    "    return filtered_data\n",
    "\n",
    "ci95dairy_biogas_ic = filter_confidence_interval_ic(dairy_ic)\n",
    "\n",
    "plt.scatter(ci95dairy_biogas_ic['Dairy'], ci95dairy_biogas_ic['Biogas_gen_ft3_day'])\n",
    "\n",
    "dairy_biogas6 = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=ci95dairy_biogas_ic).fit()\n",
    "\n",
    "print(dairy_biogas6.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below is a summary of efficiencies for dairy anaerobic digesters based on the AgSTAR data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Efficiencies of Digesters using 95% CI are:\n",
      "For All Dairy\n",
      "68.8173385176075\n",
      "For plug flow\n",
      "73.74554076333106\n",
      "For complete mix\n",
      "85.9351730367076\n",
      "For impermeable cover\n",
      "43.108970927491335\n"
     ]
    }
   ],
   "source": [
    "print(\"Efficiencies of Digesters using 95% CI are:\") \n",
    "print(\"For All Dairy\") \n",
    "\n",
    "print(efficiency(ci95_df2['Biogas_ft3/cow'].mean())) \n",
    "\n",
    "print(\"For plug flow\")\n",
    "\n",
    "print(efficiency(ci95_df3['Biogas_ft3/cow'].mean())) \n",
    "\n",
    "print(\"For complete mix\") \n",
    "\n",
    "print(efficiency(ci95_df4['Biogas_ft3/cow'].mean())) \n",
    "\n",
    "print(\"For impermeable cover\")\n",
    "\n",
    "print(efficiency(ci95_df5['Biogas_ft3/cow'].mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Efficiencies of Digesters using 68% CI (1 standard deviation) are:\n",
      "For All Dairy\n",
      "65.78025898722731\n",
      "For plug flow\n",
      "71.27049895301684\n",
      "For complete mix\n",
      "82.74672041445051\n",
      "For impermeable cover\n",
      "41.088176141031695\n"
     ]
    }
   ],
   "source": [
    "print(\"Efficiencies of Digesters using 68% CI (1 standard deviation) are:\") \n",
    "print(\"For All Dairy\") \n",
    "\n",
    "print(efficiency(ci68_df2['Biogas_ft3/cow'].mean())) \n",
    "\n",
    "print(\"For plug flow\")\n",
    "\n",
    "print(efficiency(ci68_df3['Biogas_ft3/cow'].mean())) \n",
    "\n",
    "print(\"For complete mix\") \n",
    "\n",
    "print(efficiency(ci68_df4['Biogas_ft3/cow'].mean())) \n",
    "\n",
    "print(\"For impermeable cover\")\n",
    "\n",
    "print(efficiency(ci68_df5['Biogas_ft3/cow'].mean()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Complete mix and plug flow anaerobic digester types for dairy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Complete mix and plug flow anaerobic digesters are heated and impermeable cover anaerobic digesters are not. I analyzed complete mix and plug anaerobic digesters together to calculate an efficiency for heated anaerobic digesters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "df6 = df.rename(columns={\"Animal/Farm Type(s)\" : \"Animal\", \"Co-Digestion\" : \"Codigestion\", \"Biogas End Use(s)\" : \"Biogas_End_Use\", \" Biogas Generation Estimate (cu_ft/day) \" : \"Biogas_gen_ft3_day\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "df6.drop(df6[(df6['Animal'] != 'Dairy')].index, inplace = True)\n",
    "df6.drop(df6[(df6['Codigestion'] != 0)].index, inplace = True)\n",
    "df6.drop(df6[(df6['Biogas_gen_ft3_day'] == 0)].index, inplace = True)\n",
    "df6['Biogas_ft3/cow'] = df6['Biogas_gen_ft3_day'] / df6['Dairy']\n",
    "\n",
    "#df3.drop(df3[(df3['Biogas_End_Use'] == 0)].index, inplace = True)\n",
    "\n",
    "#selecting for 'Complete Mix Mini Digester', 'Complete Mix','Vertical Plug Flow', 'Horizontal Plug Flow', and 'Plug Flow - Unspecified', 'Modular Plug Flow', 'Mixed Plug FLow'\n",
    "\n",
    "notwant = ['Covered Lagoon', 'Unknown or Unspecified', 0,\n",
    "       'Fixed Film/Attached Media',\n",
    "       'Primary digester tank with secondary covered lagoon',\n",
    "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor', 'Dry Digester', \n",
    "       'Microdigester']\n",
    "\n",
    "df6 = df6[~df6['Digester Type'].isin(notwant)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Biogas_ft3/cow', ylabel='Count'>"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data = df6['Biogas_ft3/cow'], bins = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "ci95_df6 = hist_filter_ci(df6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "77.10390425281459"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci95_df6['Biogas_ft3/cow'].mean()\n",
    "efficiency(ci95_df6['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "75.29627474795387"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci68_df6  = hist_filter_ci_68(df6)\n",
    "efficiency(ci68_df6['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(46, 22)"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df6.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.820\n",
      "Model:                            OLS   Adj. R-squared:                  0.816\n",
      "Method:                 Least Squares   F-statistic:                     201.1\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           5.13e-18\n",
      "Time:                        16:13:11   Log-Likelihood:                -607.56\n",
      "No. Observations:                  46   AIC:                             1219.\n",
      "Df Residuals:                      44   BIC:                             1223.\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept  -2.482e+04   2.57e+04     -0.968      0.338   -7.65e+04    2.69e+04\n",
      "Dairy        107.9092      7.610     14.180      0.000      92.572     123.246\n",
      "==============================================================================\n",
      "Omnibus:                       22.878   Durbin-Watson:                   1.873\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):               53.726\n",
      "Skew:                           1.252   Prob(JB):                     2.16e-12\n",
      "Kurtosis:                       7.665   Cond. No.                     4.35e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 4.35e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "dairy_plugflow = pd.DataFrame(df6,columns=['Dairy', \"Biogas_gen_ft3_day\"])\n",
    "\n",
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_plugflow, ci = 95)\n",
    "dairy_plugsum = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=dairy_plugflow).fit()\n",
    "print(dairy_plugsum.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<function matplotlib.pyplot.show(close=None, block=None)>"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.regplot('Dairy', 'Biogas_gen_ft3_day', data=dairy_plugflow, ci = 95)\n",
    "\n",
    "Y_upper_dairy_biogas_pf = dairy_plugflow['Dairy']*123.246+.000269\n",
    "Y_lower_dairy_biogas_pf = dairy_plugflow['Dairy']*92.572-.000765\n",
    "\n",
    "plt.scatter(dairy_plugflow['Dairy'], dairy_plugflow['Biogas_gen_ft3_day'])\n",
    "plt.scatter(dairy_plugflow['Dairy'], Y_upper_dairy_biogas_pf, color = 'red')\n",
    "plt.scatter(dairy_plugflow['Dairy'], Y_lower_dairy_biogas_pf, color = 'red')\n",
    "\n",
    "plt.show"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "def filter_confidence_interval_pf(data):\n",
    "    Y_upper = data['Dairy']*123.246+.000269\n",
    "    Y_lower = data['Dairy']*92.572-.000765\n",
    "    filtered_data = data[(data['Biogas_gen_ft3_day'] >= Y_lower) & (data['Biogas_gen_ft3_day'] <= Y_upper)]\n",
    "    return filtered_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x1b8654a2dc0>"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "ci95dairy_biogas_pf = filter_confidence_interval_pf(dairy_plugflow)\n",
    "\n",
    "plt.scatter(ci95dairy_biogas_pf['Dairy'], ci95dairy_biogas_pf['Biogas_gen_ft3_day'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:     Biogas_gen_ft3_day   R-squared:                       0.999\n",
      "Model:                            OLS   Adj. R-squared:                  0.998\n",
      "Method:                 Least Squares   F-statistic:                     4873.\n",
      "Date:                Sun, 30 Apr 2023   Prob (F-statistic):           3.26e-11\n",
      "Time:                        16:13:12   Log-Likelihood:                -98.196\n",
      "No. Observations:                   9   AIC:                             200.4\n",
      "Df Residuals:                       7   BIC:                             200.8\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept   7996.2535   6143.898      1.301      0.234   -6531.756    2.25e+04\n",
      "Dairy         99.8207      1.430     69.806      0.000      96.439     103.202\n",
      "==============================================================================\n",
      "Omnibus:                       14.718   Durbin-Watson:                   1.972\n",
      "Prob(Omnibus):                  0.001   Jarque-Bera (JB):                6.481\n",
      "Skew:                           1.783   Prob(JB):                       0.0392\n",
      "Kurtosis:                       5.137   Cond. No.                     5.27e+03\n",
      "==============================================================================\n",
      "\n",
      "Notes:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 5.27e+03. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\scipy\\stats\\stats.py:1541: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=9\n",
      "  warnings.warn(\"kurtosistest only valid for n>=20 ... continuing \"\n"
     ]
    }
   ],
   "source": [
    "dairy_biogas4 = smf.ols(formula='Biogas_gen_ft3_day ~ Dairy', data=ci95dairy_biogas_pf).fit()\n",
    "print(dairy_biogas4.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Swine Impermeable Cover Investigation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The data for the swine anaerobic digesters was very poor. I decided to only analyze impermeable cover digester type for swine as I felt the other digester types had too little data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "df7 = df.rename(columns={\"Animal/Farm Type(s)\" : \"Animal\", \"Co-Digestion\" : \"Codigestion\", \"Biogas End Use(s)\" : \"Biogas_End_Use\", \" Biogas Generation Estimate (cu_ft/day) \" : \"Biogas_gen_ft3_day\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "df7.drop(df7[(df7['Animal'] != 'Swine')].index, inplace = True)\n",
    "df7.drop(df7[(df7['Codigestion'] != 0)].index, inplace = True)\n",
    "df7.drop(df7[(df7['Biogas_gen_ft3_day'] == 0)].index, inplace = True)\n",
    "df7['Biogas_ft3/cow'] = df7['Biogas_gen_ft3_day'] / df7['Swine']\n",
    "\n",
    "#df7.drop(df7[(df7['Biogas_End_Use'] == 0)].index, inplace = True)\n",
    "\n",
    "#selecting for 'Covered Lagoon'\n",
    "\n",
    "notwant = ['Mixed Plug Flow', 'Unknown or Unspecified',\n",
    "       'Complete Mix', 'Horizontal Plug Flow', 0,\n",
    "       'Fixed Film/Attached Media',\n",
    "       'Primary digester tank with secondary covered lagoon',\n",
    "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor',\n",
    "       'Vertical Plug Flow', 'Complete Mix Mini Digester',\n",
    "       'Plug Flow - Unspecified', 'Dry Digester', 'Modular Plug Flow',\n",
    "       'Microdigester']\n",
    "\n",
    "df7 = df7[~df7['Digester Type'].isin(notwant)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Biogas_ft3/cow', ylabel='Count'>"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data = df7['Biogas_ft3/cow'], bins = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8, 22)"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df7.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "ci95_df7 = hist_filter_ci(df7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Biogas_ft3/cow', ylabel='Count'>"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(data = ci95_df7['Biogas_ft3/cow'], bins = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.725326735990197"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ci95_df7['Biogas_ft3/cow'].mean()\n",
    "efficiency(ci95_df7['Biogas_ft3/cow'].mean())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Checking if there was some correlation between location and impermeable cover digester efficiency. There does not appear to be the desired correlation. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Swine', ylabel='Biogas_gen_ft3_day'>"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZ0AAAEGCAYAAAC+fkgiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAkYUlEQVR4nO3de3hU1b3/8fdXEghyk6sCQSMKVkMpl9SDx1JRtF5qq1RFOMeK1panyrFg7UV+fVqxP221pUrxPPWUlla0FgSOnqr9oVIv53haRQMIcjGCihoIEFG5BxP8/v7YKziBkMyQmT2T5PN6nnlmz3fvtec7AfJlr71mLXN3RERE4nBUthMQEZHWQ0VHRERio6IjIiKxUdEREZHYqOiIiEhs8rKdQDb06NHDi4qKsp2GiEizsnTp0vfdvWdTztEqi05RURGlpaXZTkNEpFkxs3eaeg51r4mISGxUdEREJDYqOiIiEptWeU9HRCQTqqurKS8vp6qqKtupNElBQQGFhYXk5+en/dwqOiIiaVJeXk6nTp0oKirCzLKdzhFxd7Zt20Z5eTknnnhi2s+voiMiaeHuvP7B66z7aB0FbQo4tdup9OvcL9tpxaqqqqpZFxwAM6N79+5UVlZm5PwqOiKSFku3LGXi4olUf1INQL9O/bhv9H2c0OWELGcWr+ZccGpl8jNoIIGINFlVTRX3rbjvQMEBeG/neyzfujyLWUkuUtERkSarqqmifGf5IfGte7dmIZvm54477qC4uJjBgwczZMgQlixZwowZM9izZ0+jbZM9Lleo6IhIkx1TcAyXDbzskPiQXkPiT6aZefHFF3niiSdYtmwZK1eu5G9/+xv9+vVT0RERachXT/oq1w26joI2BfQ6uhd3ffEuBvcYnO20cl5FRQU9evSgXbt2APTo0YOFCxeyadMmzj77bM4++2wArr/+ekpKSiguLubWW28FYObMmYcc9/TTT3PGGWcwbNgwrrjiCnbt2pWdD3Y47t7qHsOHD3cRSb/9n+z3il0V/v6e97OdSlasWbMm5TY7d+70z33ucz5gwAC//vrr/fnnn3d39xNOOMErKysPHLdt2zZ3d6+pqfGzzjrLV6xYcchxlZWVPnLkSN+1a5e7u995551+2223pe2zAKXexN+/Gr0mImlzlB3FcR2Oy3YazUrHjh1ZunQpL7zwAs899xxXXnkld9555yHHzZ8/n1mzZlFTU0NFRQVr1qxh8OC6V5IvvfQSa9as4cwzzwTg448/5owzzojlcyRLRUdEJMvatGnDqFGjGDVqFJ/97GeZM2dOnf1vv/0206dP55VXXqFr165cc8019c564O6cd955zJ07N67UU6Z7OiIiWVRWVsa6desOvH711Vc54YQT6NSpEzt37gRgx44ddOjQgS5durBlyxYWLVp04PjE40aMGMHf//531q9fD8CePXt44403Yvw0jdOVjohIFu3atYsbb7yRjz76iLy8PE4++WRmzZrF3LlzufDCC+nduzfPPfccQ4cOpbi4mP79+x/oPgOYOHFinePuv/9+xo8fz759+wC4/fbbGThwYLY+3iEsujfUupSUlLgWcRORdFu7di2nnnpqttNIi/o+i5ktdfeSppxX3WsiIhIbFR0REYmNio6IiMRGRUdERGKjoiMiIrFR0RERkdio6IiItCAdO3YEYMOGDQwaNCjL2RxKRUdERGKjGQlERLLkv5Zv5JdPlbHpo730OaY93z//FC4d2jfbaWWUio6ISBb81/KNTH3kNfZW7wdg40d7mfrIawAtuvCoe01EJAt++VTZgYJTa2/1fn75VFmWMoqHio6ISBZs+mhvSvGWQkVHRCQL+hzTPqV4S6GiIyKSBd8//xTa57epE2uf34bvn39K2t6jrKyMwsLCA48FCxak7dxHSgMJRESyoHawQLpHr+3atQuAoqIiqqurm5xnuqnoiIhkyaVD+7bokWr1UfeaiIjERkVHRERio6IjIiKxUdEREZHYqOiIiEhsVHRERFoQM+Pmm28+8Hr69OlMmzbtwOsHHniAQYMGUVxczGmnncb06dNjzU9FR0SkBWnXrh2PPPII77///iH7Fi1axIwZM3j66adZvXo1y5Yto0uXLrHml/GiY2Y3mdlqM1tlZnPNrMDMupnZYjNbF567Jhw/1czWm1mZmZ2fEB9uZq+FfTPNzEK8nZk9HOJLzKwo059JRCQtVs6HewbBtGOi55Xzm3zKvLw8Jk6cyD333HPIvp///OdMnz6dPn36AFBQUMC3vvWtJr9nKjJadMysL/AdoMTdBwFtgHHALcAz7j4AeCa8xsxOC/uLgQuA35hZ7TwR9wETgQHhcUGIXwd86O4nA/cAd2XyM4mIpMXK+fD4d2D7e4BHz49/Jy2FZ9KkSTz00ENs3769TnzVqlUMHz68yedviji61/KA9maWBxwNbAIuAeaE/XOAS8P2JcA8d9/n7m8D64HTzaw30NndX3R3Bx44qE3tuRYCo2uvgkREctYzP4Xqg2aUrt4bxZuoc+fOXH311cycObPJ50q3jBYdd98ITAfeBSqA7e7+NHCsu1eEYyqAXqFJX+C9hFOUh1jfsH1wvE4bd68BtgPdD87FzCaaWamZlVZWVqbnA4qIHKnt5anFUzRlyhRmz57N7t27D8SKi4tZunRpWs5/pDLdvdaV6ErkRKAP0MHMrmqoST0xbyDeUJu6AfdZ7l7i7iU9e/ZsOHERkUzrUphaPEXdunVj7NixzJ49+0Bs6tSp/OAHP2Dz5s0A7Nu3L/aroUx3r50LvO3ule5eDTwC/DOwJXSZEZ63huPLgX4J7QuJuuPKw/bB8TptQhdeF+CDjHwaEZF0Gf0TyD9o7Zz89lE8TW6++eY6o9guuugiJk2axLnnnktxcTHDhw+npqYmbe+XjEzPMv0uMMLMjgb2AqOBUmA3MAG4Mzz/JRz/GPBnM7ub6MpoAPCyu+83s51mNgJYAlwN3JvQZgLwInA58Gy47yMikrsGj42en/lp1KXWpTAqOLXxI1S7tAHAsccey549e+rsv/baa7n22mub9B5NkdGi4+5LzGwhsAyoAZYDs4COwHwzu46oMF0Rjl9tZvOBNeH4Se5eu4j49cD9QHtgUXgAzAYeNLP1RFc44zL5mURE0mbw2CYXmeYm4+vpuPutwK0HhfcRXfXUd/wdwB31xEuBQfXEqwhFS0REcptmJBARkdio6IiISGxUdEREJDYqOiIiEhsVHRGRFmbz5s2MGzeOk046idNOO42LLrqIN954A4B77rmHgoKCQ+Zli4uKjohIC+LujBkzhlGjRvHmm2+yZs0afvazn7FlyxYA5s6dy+c//3keffTRrOSnoiMikiV/feuvfGnhlxg8ZzBfWvgl/vrWX5t8zueee478/Hy+/e1vH4gNGTKEkSNH8uabb7Jr1y5uv/125s6d2+T3OhIqOiIiWfDXt/7KtH9Mo2J3BY5TsbuCaf+Y1uTC09DyBXPnzmX8+PGMHDmSsrIytm7dWu9xmaSiIyKSBb9e9muq9lfViVXtr+LXy36dsfecN28e48aN46ijjuJrX/saCxYsyNh7HU7GZyQQEZFDbd69OaV4soqLi1m4cOEh8ZUrV7Ju3TrOO+88AD7++GP69+/PpEmTmvR+qdKVjohIFhzX4biU4sk655xz2LdvH7/73e8OxF555RUmT57MtGnT2LBhAxs2bGDTpk1s3LiRd955p0nvl6qki46ZdctkIiIircnkYZMpaFNQJ1bQpoDJwyY36bxmxqOPPsrixYs56aSTKC4uZtq0aTz//POMGTOmzrFjxoxh3rx5TXq/VKXSvbbEzF4F/ggs0vIBIiJH7sv9vwxE93Y2797McR2OY/KwyQfiTdGnTx/mz5/f6HF33313k98rVakUnYFEi7J9A7jXzB4G7nf3NzKSmYhIC/fl/l9OS5FpTpLuXvPIYncfD3yTaOG0l83sv83sjIxlKCIiLUbSVzpm1h24Cvg6sAW4kWjVziHAAuDEDOQnIiItSCrday8CDwKXunt5QrzUzP4jvWmJiEhLlErROeVwgwfc/a405SMiIi1YKkWnh5n9ACgGDozzc/dz0p6ViIi0SKl8OfQh4HWieze3ARuAVzKQk4iIHIFRo0bx1FNP1YnNmDGDG264gcrKSvLz8/ntb3+bpewiqRSd7u4+G6h29/92928AIzKUl4iIpGj8+PGHfNlz3rx5jB8/ngULFjBixIiszS5dK5WiUx2eK8zsy2Y2FCjMQE4iIq3C9scfZ905o1l76mmsO2c02x9/vEnnu/zyy3niiSfYt28fwIHpbr7whS8wd+5cfvWrX1FeXs7GjRvTkf4RSaXo3G5mXYCbge8BvwduykhWIiIt3PbHH6fixz+hZtMmcKdm0yYqfvyTJhWe7t27c/rpp/Pkk08C0VXOlVdeSXl5OZs3b+b0009n7NixPPzww+n6GClL5cuhT7j7dndf5e5nu/twd38sk8mJiLRUW++ZgVfVXdrAq6rYes+MJp03sYuttmtt3rx5jB07FoBx48ZltYut0dFrZnYvcNh51tz9O2nNSESkFaipqEgpnqxLL72U7373uyxbtoy9e/cybNgwvvnNb7JlyxYeeughADZt2sS6desYMGBAk97rSCRzpVMKLCUaJj0MWBceQ4D9GctMRKQFy+vdO6V4sjp27MioUaP4xje+wfjx4ykrK2P37t1s3LjxwLIGU6dOjX126VqNFh13n+Puc4ABwNnufq+73wuMJio8IiKSol43TcEK6i5tYAUF9LppSpPPPX78eFasWHGgK+3gJQ0uu+yyrHWxpfLl0D5AJ+CD8LpjiImISIq6fOUrQHRvp6aigrzevel105QD8aYYM2YMtRPITJs27ZD9gwcPZs2aNU1+nyORStG5E1huZs+F12cB09KekYhIK9HlK19JS5FpTpIuOu7+RzNbBPxTCN3i7gcW8zazYndfne4ERUSk5UjlSodQZP5ymN0PEg00EBFptdwdM8t2Gk2SyYWhU/lyaGOa909ZRKSJCgoK2LZtW0Z/aWeau7Nt2zYKDhrkkC4pXek0ovn+lEVE0qCwsJDy8nIqKyuznUqTFBQUUFiYmVnO0ll0RERatfz8fE48UYsoNySd3Wsfp/FcIiLSAiVVdMzsKDM7Kmy3NbNhZtYt8Rh31zIHIiLSoEaLjpldClQAG83sEuAFYDqw0sxa1wBzERFpkmTu6dwKfA5oD6wAPu/uZWZ2AvCfQNMWgBARkVYjqYEEtV8CNbN33b0sxN6p7XITERFJRtL3dMLmNxJibYC2SbQ9xswWmtnrZrbWzM4ws25mttjM1oXnrgnHTzWz9WZWZmbnJ8SHm9lrYd9MC9++MrN2ZvZwiC8xs6IkP7uIiMQsmaIzkVBc3P3lhHg/ovnYGvNr4El3/wxRN91a4BbgGXcfADwTXmNmpwHjgGLgAuA3obgB3BdyGRAeF4T4dcCH7n4ycA9wVxI5iYhIFiSztMEr7l5lZpMPim8AujfU1sw6A18EZoc2H7v7R8AlwJxw2Bzg0rB9CTDP3fe5+9vAeuB0M+sNdHb3Fz36qu8DB7WpPddCYHTtVZCIiOSWVO7JTKgndk0jbfoDlcAfzWy5mf3ezDoAx7p7BUB47hWO7wu8l9C+PMT6hu2D43XauHsNsJ16iqGZTTSzUjMrbe7fFhYRaa6SWa56PPAvQH8zeyxhVydgWxLnHwbc6O5LzOzXhK60w71dPTFvIN5Qm7oB91nALICSkhJN2SMikgXJjF57ieh7Oj2AXyXEdwIrG2lbDpS7+5LweiFR0dliZr3dvSJ0nW1NOL5fQvtCYFOIF9YTT2xTbmZ5QBc+XWhORERySDLdawvd/Xlgj7v/d8JjWejOOqww1Po9MzslhEYDa4DH+LS7bgKfLpfwGDAujEg7kWjAwMuhC26nmY0I92uuPqhN7bkuB5715jzFq4hIC5bMlc5RZnYrMNDMvnvwTne/u5H2NwIPmVlb4C3gWqJiN9/MrgPeBa4I51ptZvOJClMNMMnd94fzXA/cT/Ql1UXhAdEghQfNbD3RFc64JD6TiIhkQTJFZxzRSLE8ovs4KXH3V4GSenaNPszxdwB31BMvBQbVE68iFC0REcltjRadMAPBXWa20t0XHe44M5vg7nMOt19ERCTpIdMNFZxgciP7RUSkldNy1SIiEpt0Fh2NGBMRkQbpSkdERGKTzqLz9zSeS0REWqCk1tOBaAkB4DKgKLGdu/80PP9bupMTEZGWJemiQzQDwHZgKbAvM+mIiEhLlkrRKXT3Cxo/TEREpH6p3NP5h5l9NmOZiIhIi5fKlc4XgGvM7G2i7jUD3N0HZyQzERFpcVIpOhdmLAsREWkVUpkG5x2idWvOCdt7UmkvIiKSdNEIyxv8EJgaQvnAnzKRlIhITqjeC9VV2c6iRUmle20MMBRYBuDum8ws5aUORERyXtUOePNZ+MdMyD8azpwCRSMhv122M2v2Uik6H7u7m5kDmFmHDOUkIpJdbz0PCyZ8+nrDCzDhCThxZNZSailSuScz38x+CxxjZt8C/gb8LjNpiYhkyf6P4aX7Do2veSz+XFqgpK903H26mZ0H7ABOAX7i7oszlpmISFYcBQVdDg0X6G5COqTSvUYoMio0ItJytcmDMybBuqfAP4lieQXwmYuzm1cLkcqEnzs5dM2c7UApcLO7v5XOxEREsub4EXDtInjjKchvDwPOg95Dsp1Vi5DKlc7dwCbgz0SzEYwDjgPKgD8Ao9KdnIhIVrTJjwrP8SOynUmLk8pAggvc/bfuvtPdd7j7LOAid38Y6Jqh/EREpAVJpeh8YmZjzeyo8BibsE9LVYuISKNSKTr/Cnwd2ApsCdtXmVl7QAu4iYhIo1IZMv0W8JXD7P5fM5vq7j9PT1oiItISpXPCzivSeC4REWmB0ll0LI3nEhGRFiidRUeDCUREpEG60hERkdiks+gsSOO5RESkBUplEbdfmFlnM8s3s2fM7H0zu6p2v7v/LDMpiohIS5HKlc6X3H0HcDFQDgwEvp+RrEREpEVKpejkh+eLgLnu/kEG8hERkRYslQk/Hzez14G9wA1m1hPQ4uEiIpK0VGYkuMXM7gJ2uPt+M9sNXJK51EREJC2q98KW1fDRu9DpODh2EBR0zkoqKS3iBvQFzjOzgoTYA2nMR0RE0umTT2DFPHhiyqexkTfDyO9B26NjTyeV0Wu3AveGx9nAL4CvZigvERFJhw/ehCdvqRt74VdQ+XpW0kllIMHlwGhgs7tfC3wOaJeRrEREJD2qtkNNPbff92RnLFgqRWevu38C1JhZZ6IlDvpnJi0REUmLLoXQpV/dWNsO0LUoK+mkUnRKzewY4HfAUmAZ8HIyDc2sjZktN7MnwutuZrbYzNaF564Jx041s/VmVmZm5yfEh5vZa2HfTDOzEG9nZg+H+BIzK0rhM4mItGydjoMrH4wGDwAcUwTj50GPk7OSjrmnPk9n+MXe2d1XJnn8d4GS0OZiM/sF8IG732lmtwBd3f2HZnYaMBc4HegD/A0YGEbLvQxMBl4C/h8w090XmdkNwGB3/7aZjQPGuPuVDeVTUlLipaWlKX9uEZFma8+HsHsrtO8KHXsd0SnMbKm7lzQljVQGEgyrfQDdgDwzO8nMGhwBZ2aFwJeB3yeELwHmhO05wKUJ8Xnuvs/d3wbWA6ebWW+igvWiR1XygYPa1J5rITC69ipIRESCo7tCz1OOuOCkSypDpn8DDANWEs0oPShsdzezb7v704dpNwP4AdApIXasu1cAuHuFmdX+FPoSXcnUKg+x6rB9cLy2zXvhXDVmth3oDryfmISZTQQmAhx//PHJfWIREUmrVO7pbACGunuJuw8HhgKrgHOJhk8fwswuBra6+9Ik36O+KxRvIN5Qm7oB91kh95KePXsmmY6IiKRTKlc6n3H31bUv3H2NmQ1197ca6M06E/iqmV0EFACdzexPwBYz6x2ucnoTjYSD6AomcZhFIbApxAvriSe2KQ9dfV0AzQsnIpKDUrnSKTOz+8zsrPD4DfCGmbUj6v46hLtPdfdCdy8CxgHPuvtVwGPAhHDYBOAvYfsxYFwYkXYiMAB4OXTF7TSzEeF+zdUHtak91+XhPbSKqYhIDkrlSuca4AZgClGX1v8C3yMqOGen+L53AvPN7DrgXeAKAHdfbWbzgTVADTDJ3feHNtcD9wPtgUXhATAbeNDM1hNd4YxLMRcREYlJSkOmzawtcArRPZMyd6/3CifXaci0iEjq0jFkOukrHTMbRTQ0eQPRlU4/M5vg7v/TlARERJq7mv2f8GblbjZ+uIcendoxoFdH2rdNdT7l1iGVn8qviFYPLQMws4FEX+QcnonERESai8Vrt3Djn5dT80nUc/TDC07h2jOLKMhX4TlYSiuH1hYcAHd/g09XExURaZXe+2APP1y48kDBAbjryTLe2LIri1nlrlTKcKmZzQYeDK//lWgONhGRVuvDPR+zo6rmkPj7O/dlIZvcl8qVzvXAauA7RHOgrQG+nYmkRESai2M7F9C7S0GdWN5RRt+u8S+Q1hwkXXTCfGh3u/vX3H2Mu9/j7irlItKqHdu5gHvHD+W4zlHh6VyQx73/MpSTenbIcma5qdHuNTOb7+5jzew16p9eZnBGMhMRaSZKirrx2L+dyeYdVXQ9ui39uukq53CSuaczOTxfnMlERESas16dC+jVuaDxA1u5RotOwmzQ79TGzKwHsE3TzYiISCoavacT5jt73sweMbOhZraKaHbpLWZ2QeZTFBGRliKZ7rV/B/4P0ezNzwIXuvtLZvYZoi+HPpnB/ESktdmyFlYthPJSGDwWTh4dLbksLUIyRSevdoE2M/upu78E4O6va4FOEUmrDzfAn8bAzoro9dvPwz9/B0bfCm307f6WIJkh058kbO89aJ/u6YhI+mxZ82nBqbXkPvjo3ezkI2mXzH8dPmdmO4gm+WwftgmvNVRDRDLM6l8fWJqlZEavtYkjERERjj0NOvWBnZs+jY24Hrocn72cJK3USSoiuaNrEXz9EVj1KGxaBp+9DPqfo/s5LYj+JEUkt/Q6Fc45NdtZSIakMuGniIhIk6joiIhIbFR0REQkNio6IiISGxUdERGJjYqOiIjERkVHRERio6IjIiKxUdEREZHYqOiIiEhsVHRERCQ2KjoiIhIbFR0REYmNio6IiMRGRUdERGKjoiMiIrFR0RERkdio6IiISGxUdEREJDYqOiIiEhsVHRERiY2KjoiIxCajRcfM+pnZc2a21sxWm9nkEO9mZovNbF147prQZqqZrTezMjM7PyE+3MxeC/tmmpmFeDszezjEl5hZUSY/k4iIHLlMX+nUADe7+6nACGCSmZ0G3AI84+4DgGfCa8K+cUAxcAHwGzNrE851HzARGBAeF4T4dcCH7n4ycA9wV4Y/k4iIHKGMFh13r3D3ZWF7J7AW6AtcAswJh80BLg3blwDz3H2fu78NrAdON7PeQGd3f9HdHXjgoDa151oIjK69ChIRkdwS2z2d0O01FFgCHOvuFRAVJqBXOKwv8F5Cs/IQ6xu2D47XaePuNcB2oHs97z/RzErNrLSysjJNn0pERFIRS9Exs47AfwJT3H1HQ4fWE/MG4g21qRtwn+XuJe5e0rNnz8ZSFhGRDMh40TGzfKKC85C7PxLCW0KXGeF5a4iXA/0SmhcCm0K8sJ54nTZmlgd0AT5I/ycREZGmyvToNQNmA2vd/e6EXY8BE8L2BOAvCfFxYUTaiUQDBl4OXXA7zWxEOOfVB7WpPdflwLPhvo+IiOSYvAyf/0zg68BrZvZqiP0f4E5gvpldB7wLXAHg7qvNbD6whmjk2yR33x/aXQ/cD7QHFoUHREXtQTNbT3SFMy7Dn0lERI6QtcaLgpKSEi8tLc12GiIizYqZLXX3kqacQzMSiIhIbFR0REQkNio6IiISGxUdERGJjYqOiIjERkVHRERio6IjIiKxUdEREZHYqOiIiEhsVHRS4NXV7N9ble00RESarUzPvdYiuDt7ly9n2x/+QPXGTXT9l/F0Omc0ed27ZTs1EZFmRUUnCVVr1/LuhGvw6moANv/4J3yyZw/dJ0xopKWIiCRS91oSqlatPlBwan3w+9lUV76fpYxERJonFZ0kWNu2h8YKCrC8NlnIRkSk+VLRSUL7zw6iTdeudWI9b5pC3kExERFpmO7pJKHdSSdx/Jz72fXCC9Rs3kLHUWfRfujQbKclItLsqOgkqWDgQAoGDsx2GiIizZq610REJDYqOiIiEhsVHRERiY2KjoiIxEZFR0REYqOiIyIisTF3z3YOsTOzSuCdI2zeA2hu898o53g0t5ybW76gnONyuJxPcPeeTTlxqyw6TWFmpe5eku08UqGc49Hccm5u+YJyjksmc1b3moiIxEZFR0REYqOik7pZ2U7gCCjneDS3nJtbvqCc45KxnHVPR0REYqMrHRERiY2KjoiIxKbVFx0zKzCzl81shZmtNrPbQrybmS02s3XhuWtCm6lmtt7Myszs/IT4cDN7LeybaWaW4dzbmNlyM3uiOeRsZhvCe71qZqXNJOdjzGyhmb1uZmvN7IxcztnMTgk/39rHDjObkuM53xT+7a0ys7nh32TO5hvea3LId7WZTQmxnMrZzP5gZlvNbFVCLG05mlk7M3s4xJeYWVFSibl7q34ABnQM2/nAEmAE8AvglhC/BbgrbJ8GrADaAScCbwJtwr6XgTPCORcBF2Y49+8CfwaeCK9zOmdgA9DjoFiu5zwH+GbYbgsck+s5J+TeBtgMnJCrOQN9gbeB9uH1fOCaXM03vM8gYBVwNNGaZH8DBuRazsAXgWHAqkz8ewNuAP4jbI8DHk4qr0z/xW9Oj/CXaBnwT0AZ0DvEewNlYXsqMDWhzVPhD6Q38HpCfDzw2wzmWgg8A5zDp0Un13PewKFFJ2dzBjoT/UK05pLzQXl+Cfh7LudMVHTeA7oR/QJ/IuSdk/mGc18B/D7h9Y+BH+RizkARdYtO2nKsPSZs5xHNYGCN5dTqu9fgQDfVq8BWYLG7LwGOdfcKgPDcKxxe+4+kVnmI9Q3bB8czZQbRX/RPEmK5nrMDT5vZUjOb2Axy7g9UAn+0qBvz92bWIcdzTjQOmBu2czJnd98ITAfeBSqA7e7+dK7mG6wCvmhm3c3saOAioF+O51wrnTkeaOPuNcB2oHtjCajoAO6+392HEF09nG5mgxo4vL4+V28gnnZmdjGw1d2XJtuknlisOQdnuvsw4EJgkpl9sYFjcyHnPKLuifvcfSiwm6hL4nByIecoEbO2wFeBBY0dWk8stpzDPYVLiLp0+gAdzOyqhpocJq/Yfsbuvha4C1gMPEnULVXTQJOs55yEI8nxiPJX0Ung7h8BzwMXAFvMrDdAeN4aDisn+l9NrUJgU4gX1hPPhDOBr5rZBmAecI6Z/SnHc8bdN4XnrcCjwOk5nnM5UB6ufAEWEhWhXM651oXAMnffEl7nas7nAm+7e6W7VwOPAP+cw/kC4O6z3X2Yu38R+ABYl+s5B+nM8UAbM8sDuhD9LBrU6ouOmfU0s2PCdnuifwSvA48BE8JhE4C/hO3HgHFh5MaJRDcQXw6XqjvNbEQY3XF1Qpu0cvep7l7o7kVEXSjPuvtVuZyzmXUws06120T99qtyOWd33wy8Z2anhNBoYE0u55xgPJ92rdXmlos5vwuMMLOjw/uMBtbmcL4AmFmv8Hw88DWin3VO55yQS7pyTDzX5US/hxq/UsvEjbbm9AAGA8uBlUS/BH8S4t2JbtSvC8/dEtr8iGh0RxkJo02AknCON4F/J4mbamnIfxSfDiTI2ZyJ7o+sCI/VwI9yPefwXkOA0vD347+Ars0g56OBbUCXhFjO5gzcRvQfvVXAg0QjqHI23/BeLxD9B2QFMDoXf8ZEhbACqCa6KrkunTkCBUTdt+uJRrj1TyYvTYMjIiKxafXdayIiEh8VHRERiY2KjoiIxEZFR0REYqOiIyIisVHREckQM/tRmIV4pUUzPv9TEm1+ambnxpGfSDZoyLRIBpjZGcDdwCh332dmPYC2HmZlEGmtdKUjkhm9gffdfR+Au78PFJrZIwBmdomZ7TWzthatH/NWiN9vZpeH7Q1mdpuZLQvrmXwmxDtYtFbKK2Ei0kuy8xFFUqeiI5IZTwP9zOwNM/uNmZ1FtGzG0LB/JNG3vD9PtJTGkvpPw/seTZJ6H/C9EPsR0ZQjnwfOBn4ZphYSyXkqOiIZ4O67gOHARKLlER4GrgLWm9mpRJOd3k200NZIomlV6vNIeF5KtDYKRPPW3RKW43ieaDqS49P9GUQyIS/bCYi0VO6+n6goPG9mrxFNjvgC0QzQ1UQrTt5PtMLn9+o/C/vC834+/fdqwGXuXpaRxEUySFc6IhlgZqeY2YCE0BDgHeB/gCnAi+5eSTQB42eIJkFN1lPAjWHWX8xsaCPHi+QMXemIZEZH4N6wbEYN0Uy8E4kWgjuWqPhANHv1Vk9tGOn/JVo5dmUoPBuAi9OStUiGaci0iIjERt1rIiISGxUdERGJjYqOiIjERkVHRERio6IjIiKxUdEREZHYqOiIiEhs/j9wdtYS9IuIiQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot('Swine', 'Biogas_gen_ft3_day', data = ci95_df7, hue = 'State')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Swine Impermeable Cover and Electricity Investigation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The next section investigates \"Swine Impermeable Cover and Electricity Investigation\" because there are a variety of swine digester data points where biogas production is not recorded, although electricity generation is recorded."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [],
   "source": [
    "df8 = df.rename(columns={\"Animal/Farm Type(s)\" : \"Animal\", \"Co-Digestion\" : \"Codigestion\", \"Biogas End Use(s)\" : \"Biogas_End_Use\", \" Biogas Generation Estimate (cu_ft/day) \" : \"Biogas_gen_ft3_day\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "df8.drop(df8[(df8['Animal'] != 'Swine')].index, inplace = True)\n",
    "df8.drop(df8[(df8['Codigestion'] != 0)].index, inplace = True)\n",
    "df8.drop(df8[(df8[' Electricity Generated (kWh/yr) '] == 0)].index, inplace = True)\n",
    "df8['Elec/swine'] = df8[' Electricity Generated (kWh/yr) '] / df8['Swine']\n",
    "df8.drop(df8[(df8['Swine'] > 20000)].index, inplace = True)\n",
    "#df7.drop(df7[(df7['Biogas_End_Use'] == 0)].index, inplace = True)\n",
    "\n",
    "#selecting for 'Covered Lagoon'\n",
    "\n",
    "notwant = ['Mixed Plug Flow', 'Unknown or Unspecified',\n",
    "       'Complete Mix', 'Horizontal Plug Flow', 0,\n",
    "       'Fixed Film/Attached Media',\n",
    "       'Primary digester tank with secondary covered lagoon',\n",
    "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor',\n",
    "       'Vertical Plug Flow', 'Complete Mix Mini Digester',\n",
    "       'Plug Flow - Unspecified', 'Dry Digester', 'Modular Plug Flow',\n",
    "       'Microdigester']\n",
    "\n",
    "df8 = df8[~df8['Digester Type'].isin(notwant)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Swine', ylabel=' Electricity Generated (kWh/yr) '>"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot('Swine', ' Electricity Generated (kWh/yr) ', data = df8, hue = 'State')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Naive Bayes Machine Learning- Predict Biogas Amount and Efficiency"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Naive Bayes machine learning is completed in this section to predict the daily biogas production from an inputted number of dairy cows and an inputted digester type. Naive Bayes is a poor way to predict daily biogas production based on this data. Naive Bayes machine learns based on categories, not on a regression spectrum. For any input, Naive Bayes will output a biogas production value that corresponds to a value in the training dataset. This is not what I wanted, but I completed the analysis before I figured this out, so I have left it here."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "df9 = df.rename(columns={\"Animal/Farm Type(s)\" : \"Animal\", \"Co-Digestion\" : \"Codigestion\", \"Biogas End Use(s)\" : \"Biogas_End_Use\", \" Biogas Generation Estimate (cu_ft/day) \" : \"Biogas_gen_ft3_day\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Project Name</th>\n",
       "      <th>Project Type</th>\n",
       "      <th>City</th>\n",
       "      <th>County</th>\n",
       "      <th>State</th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Status</th>\n",
       "      <th>Year Operational</th>\n",
       "      <th>Animal</th>\n",
       "      <th>Cattle</th>\n",
       "      <th>...</th>\n",
       "      <th>Poultry</th>\n",
       "      <th>Swine</th>\n",
       "      <th>Codigestion</th>\n",
       "      <th>Biogas_gen_ft3_day</th>\n",
       "      <th>Electricity Generated (kWh/yr)</th>\n",
       "      <th>Biogas_End_Use</th>\n",
       "      <th>System Designer(s)_Developer(s) and Affiliates</th>\n",
       "      <th>Receiving Utility</th>\n",
       "      <th>Total Emission Reductions (MTCO2e/yr)</th>\n",
       "      <th>Awarded USDA Funding?</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Cargill - Sandy River Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Morrilton</td>\n",
       "      <td>Conway</td>\n",
       "      <td>AR</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2008.0</td>\n",
       "      <td>Swine</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>4200</td>\n",
       "      <td>0</td>\n",
       "      <td>1814400</td>\n",
       "      <td>0</td>\n",
       "      <td>Flared Full-time</td>\n",
       "      <td>Martin Construction Resource LLC (formerly RCM...</td>\n",
       "      <td>0</td>\n",
       "      <td>4,002</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Butterfield RNG Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Buckeye</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Mixed Plug Flow</td>\n",
       "      <td>Construction</td>\n",
       "      <td>2021.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Pipeline Gas</td>\n",
       "      <td>Avolta [Project Developer]; DVO, Inc. (formerl...</td>\n",
       "      <td>Southwest Gas</td>\n",
       "      <td>29,826</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Caballero Dairy Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Eloy</td>\n",
       "      <td>Pinal</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Unknown or Unspecified</td>\n",
       "      <td>Construction</td>\n",
       "      <td>2022.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Pipeline Gas</td>\n",
       "      <td>Brightmark [Project Developer]</td>\n",
       "      <td>0</td>\n",
       "      <td>89,518</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Paloma Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Gila Bend</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2021.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>CNG</td>\n",
       "      <td>Black Bear Environmental Assets [Project Devel...</td>\n",
       "      <td>Southwest Gas Company</td>\n",
       "      <td>89,794</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Stotz Southern Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Buckeye</td>\n",
       "      <td>Maricopa</td>\n",
       "      <td>AZ</td>\n",
       "      <td>Covered Lagoon</td>\n",
       "      <td>Operational</td>\n",
       "      <td>2011.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5256000</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Chapel Street Environmental [System Design Eng...</td>\n",
       "      <td>Arizona Public Service</td>\n",
       "      <td>138,787</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>501</th>\n",
       "      <td>Norswiss Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Rice Lake</td>\n",
       "      <td>Barron</td>\n",
       "      <td>WI</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2006.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Dairy Processing Wastes; Fats, Oils, Greases; ...</td>\n",
       "      <td>2908000</td>\n",
       "      <td>6450000</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Microgy [Project Developer, System Designer]</td>\n",
       "      <td>Dairyland Power Cooperative; Barron Electric</td>\n",
       "      <td>0</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>502</th>\n",
       "      <td>Quantum Dairy Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Weyauwega</td>\n",
       "      <td>Waupaca</td>\n",
       "      <td>WI</td>\n",
       "      <td>Mixed Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2005.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3350700</td>\n",
       "      <td>Cogeneration</td>\n",
       "      <td>DVO, Inc. (formerly GHD, Inc.) [Project Develo...</td>\n",
       "      <td>WE Energies, Inc.</td>\n",
       "      <td>0</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>503</th>\n",
       "      <td>Stencil Farm Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Denmark</td>\n",
       "      <td>Brown</td>\n",
       "      <td>WI</td>\n",
       "      <td>Horizontal Plug Flow</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2002.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>946080</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>Martin Construction Resource LLC (formerly RCM...</td>\n",
       "      <td>Wisconsin Public Service Corporation</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504</th>\n",
       "      <td>Tinedale Farms Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Wrightstown</td>\n",
       "      <td>Jackson</td>\n",
       "      <td>WI</td>\n",
       "      <td>Fixed Film/Attached Media</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>1999.0</td>\n",
       "      <td>Dairy</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>200000</td>\n",
       "      <td>5584500</td>\n",
       "      <td>Electricity; Boiler/Furnace fuel</td>\n",
       "      <td>AGES [Project Developer, System Designer, Syst...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>505</th>\n",
       "      <td>Wyoming Premium Farms 2 Digester</td>\n",
       "      <td>Farm Scale</td>\n",
       "      <td>Wheatland</td>\n",
       "      <td>Platte</td>\n",
       "      <td>WY</td>\n",
       "      <td>Complete Mix</td>\n",
       "      <td>Shut down</td>\n",
       "      <td>2004.0</td>\n",
       "      <td>Swine</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>18000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1340280</td>\n",
       "      <td>Electricity</td>\n",
       "      <td>AG Engineering, Inc. [System Design Engineer];...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>506 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                            Project Name Project Type         City    County  \\\n",
       "0    Cargill - Sandy River Farm Digester   Farm Scale    Morrilton    Conway   \n",
       "1               Butterfield RNG Digester   Farm Scale      Buckeye  Maricopa   \n",
       "2         Caballero Dairy Farms Digester   Farm Scale         Eloy     Pinal   \n",
       "3                  Paloma Dairy Digester   Farm Scale    Gila Bend  Maricopa   \n",
       "4          Stotz Southern Dairy Digester   Farm Scale      Buckeye  Maricopa   \n",
       "..                                   ...          ...          ...       ...   \n",
       "501              Norswiss Farms Digester   Farm Scale    Rice Lake    Barron   \n",
       "502               Quantum Dairy Digester   Farm Scale    Weyauwega   Waupaca   \n",
       "503                Stencil Farm Digester   Farm Scale      Denmark     Brown   \n",
       "504              Tinedale Farms Digester   Farm Scale  Wrightstown   Jackson   \n",
       "505     Wyoming Premium Farms 2 Digester   Farm Scale    Wheatland    Platte   \n",
       "\n",
       "    State              Digester Type        Status  Year Operational Animal  \\\n",
       "0      AR             Covered Lagoon   Operational            2008.0  Swine   \n",
       "1      AZ            Mixed Plug Flow  Construction            2021.0  Dairy   \n",
       "2      AZ     Unknown or Unspecified  Construction            2022.0  Dairy   \n",
       "3      AZ               Complete Mix   Operational            2021.0  Dairy   \n",
       "4      AZ             Covered Lagoon   Operational            2011.0  Dairy   \n",
       "..    ...                        ...           ...               ...    ...   \n",
       "501    WI               Complete Mix     Shut down            2006.0  Dairy   \n",
       "502    WI            Mixed Plug Flow     Shut down            2005.0  Dairy   \n",
       "503    WI       Horizontal Plug Flow     Shut down            2002.0  Dairy   \n",
       "504    WI  Fixed Film/Attached Media     Shut down            1999.0  Dairy   \n",
       "505    WY               Complete Mix     Shut down            2004.0  Swine   \n",
       "\n",
       "    Cattle  ...  Poultry  Swine  \\\n",
       "0        0  ...        0   4200   \n",
       "1        0  ...        0      0   \n",
       "2        0  ...        0      0   \n",
       "3        0  ...        0      0   \n",
       "4        0  ...        0      0   \n",
       "..     ...  ...      ...    ...   \n",
       "501      0  ...        0      0   \n",
       "502      0  ...        0      0   \n",
       "503      0  ...        0      0   \n",
       "504      0  ...        0      0   \n",
       "505      0  ...        0  18000   \n",
       "\n",
       "                                           Codigestion Biogas_gen_ft3_day  \\\n",
       "0                                                    0            1814400   \n",
       "1                                                    0                  0   \n",
       "2                                                    0                  0   \n",
       "3                                                    0                  0   \n",
       "4                                                    0                  0   \n",
       "..                                                 ...                ...   \n",
       "501  Dairy Processing Wastes; Fats, Oils, Greases; ...            2908000   \n",
       "502                                                  0                  0   \n",
       "503                                                  0                  0   \n",
       "504                                                  0             200000   \n",
       "505                                                  0                  0   \n",
       "\n",
       "      Electricity Generated (kWh/yr)                     Biogas_End_Use  \\\n",
       "0                                   0                  Flared Full-time   \n",
       "1                                   0                      Pipeline Gas   \n",
       "2                                   0                      Pipeline Gas   \n",
       "3                                   0                               CNG   \n",
       "4                             5256000                       Electricity   \n",
       "..                                ...                               ...   \n",
       "501                           6450000                       Electricity   \n",
       "502                           3350700                      Cogeneration   \n",
       "503                            946080                       Electricity   \n",
       "504                           5584500  Electricity; Boiler/Furnace fuel   \n",
       "505                           1340280                       Electricity   \n",
       "\n",
       "        System Designer(s)_Developer(s) and Affiliates  \\\n",
       "0    Martin Construction Resource LLC (formerly RCM...   \n",
       "1    Avolta [Project Developer]; DVO, Inc. (formerl...   \n",
       "2                       Brightmark [Project Developer]   \n",
       "3    Black Bear Environmental Assets [Project Devel...   \n",
       "4    Chapel Street Environmental [System Design Eng...   \n",
       "..                                                 ...   \n",
       "501       Microgy [Project Developer, System Designer]   \n",
       "502  DVO, Inc. (formerly GHD, Inc.) [Project Develo...   \n",
       "503  Martin Construction Resource LLC (formerly RCM...   \n",
       "504  AGES [Project Developer, System Designer, Syst...   \n",
       "505  AG Engineering, Inc. [System Design Engineer];...   \n",
       "\n",
       "                                Receiving Utility  \\\n",
       "0                                               0   \n",
       "1                                   Southwest Gas   \n",
       "2                                               0   \n",
       "3                           Southwest Gas Company   \n",
       "4                          Arizona Public Service   \n",
       "..                                            ...   \n",
       "501  Dairyland Power Cooperative; Barron Electric   \n",
       "502                             WE Energies, Inc.   \n",
       "503          Wisconsin Public Service Corporation   \n",
       "504                                             0   \n",
       "505                                             0   \n",
       "\n",
       "     Total Emission Reductions (MTCO2e/yr)  Awarded USDA Funding?  \n",
       "0                                     4,002                     0  \n",
       "1                                    29,826                     Y  \n",
       "2                                    89,518                     0  \n",
       "3                                    89,794                     0  \n",
       "4                                   138,787                     0  \n",
       "..                                      ...                   ...  \n",
       "501                                       0                     Y  \n",
       "502                                       0                     Y  \n",
       "503                                       0                     0  \n",
       "504                                       0                     0  \n",
       "505                                       0                     0  \n",
       "\n",
       "[506 rows x 21 columns]"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Project Name', 'Project Type', 'City', 'County', 'State',\n",
       "       'Digester Type', 'Status', 'Year Operational', 'Animal', 'Cattle',\n",
       "       'Dairy', 'Poultry', 'Swine', 'Codigestion', 'Biogas_gen_ft3_day',\n",
       "       ' Electricity Generated (kWh/yr) ', 'Biogas_End_Use',\n",
       "       'System Designer(s)_Developer(s) and Affiliates', 'Receiving Utility',\n",
       "       ' Total Emission Reductions (MTCO2e/yr) ', 'Awarded USDA Funding?'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df9.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\lcsoko\\AppData\\Local\\Temp\\ipykernel_12912\\3026402764.py:1: UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access\n",
      "  df.bayes = df9[[\"Digester Type\",\"Dairy\",\"Biogas_gen_ft3_day\"]].copy()\n"
     ]
    }
   ],
   "source": [
    "df.bayes = df9[[\"Digester Type\",\"Dairy\",\"Biogas_gen_ft3_day\"]].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Covered Lagoon', 'Mixed Plug Flow', 'Unknown or Unspecified',\n",
       "       'Complete Mix', 'Horizontal Plug Flow', 0,\n",
       "       'Fixed Film/Attached Media',\n",
       "       'Primary digester tank with secondary covered lagoon',\n",
       "       'Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor',\n",
       "       'Vertical Plug Flow', 'Complete Mix Mini Digester',\n",
       "       'Plug Flow - Unspecified', 'Dry Digester', 'Modular Plug Flow',\n",
       "       'Microdigester'], dtype=object)"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.bayes[\"Digester Type\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\lcsoko\\AppData\\Local\\Temp\\ipykernel_12912\\1780971751.py:6: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df.bayes.drop(df.bayes[(df.bayes['Biogas_gen_ft3_day'] == 0)].index, inplace = True)\n",
      "C:\\Users\\lcsoko\\AppData\\Local\\Temp\\ipykernel_12912\\1780971751.py:7: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df.bayes.drop(df.bayes[(df.bayes['Dairy'] == 0)].index, inplace = True)\n"
     ]
    }
   ],
   "source": [
    "#I want 'Covered Lagoon', 'Mixed Plug Flow','Complete Mix','Horizontal Plug Flow','Vertical Plug Flow','Plug Flow - Unspecified'\n",
    "\n",
    "notwant = ['Unknown or Unspecified', 0,'Fixed Film/Attached Media','Primary digester tank with secondary covered lagoon','Induced Blanket Reactor', 'Anaerobic Sequencing Batch Reactor', 'Complete Mix Mini Digester','Dry Digester', 'Modular Plug Flow','Microdigester']\n",
    "\n",
    "df.bayes = df.bayes[~df.bayes['Digester Type'].isin(notwant)]\n",
    "df.bayes.drop(df.bayes[(df.bayes['Biogas_gen_ft3_day'] == 0)].index, inplace = True)\n",
    "df.bayes.drop(df.bayes[(df.bayes['Dairy'] == 0)].index, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Dairy</th>\n",
       "      <th>Biogas_gen_ft3_day</th>\n",
       "      <th>biogas/dairy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15500</td>\n",
       "      <td>600000</td>\n",
       "      <td>38.709677</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1700</td>\n",
       "      <td>50000</td>\n",
       "      <td>29.411765</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>9700</td>\n",
       "      <td>270000</td>\n",
       "      <td>27.835052</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>7000</td>\n",
       "      <td>360000</td>\n",
       "      <td>51.428571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>400</td>\n",
       "      <td>30000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108</th>\n",
       "      <td>3</td>\n",
       "      <td>10000</td>\n",
       "      <td>2739726</td>\n",
       "      <td>273.972600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>2</td>\n",
       "      <td>455</td>\n",
       "      <td>14000</td>\n",
       "      <td>30.769231</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>2</td>\n",
       "      <td>380</td>\n",
       "      <td>37500</td>\n",
       "      <td>98.684211</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>3</td>\n",
       "      <td>350</td>\n",
       "      <td>33450</td>\n",
       "      <td>95.571429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112</th>\n",
       "      <td>3</td>\n",
       "      <td>1240</td>\n",
       "      <td>2908000</td>\n",
       "      <td>2345.161290</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>113 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Digester Type  Dairy  Biogas_gen_ft3_day  biogas/dairy\n",
       "0                1  15500              600000     38.709677\n",
       "1                1   1700               50000     29.411765\n",
       "2                1   9700              270000     27.835052\n",
       "3                1   7000              360000     51.428571\n",
       "4                1    400               30000     75.000000\n",
       "..             ...    ...                 ...           ...\n",
       "108              3  10000             2739726    273.972600\n",
       "109              2    455               14000     30.769231\n",
       "110              2    380               37500     98.684211\n",
       "111              3    350               33450     95.571429\n",
       "112              3   1240             2908000   2345.161290\n",
       "\n",
       "[113 rows x 4 columns]"
      ]
     },
     "execution_count": 213,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.bayes=df.bayes.reset_index(drop=True)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To simplify the machine learning, I number coded the anaerobic digester types where 1 = covered lagoon, 2 = plug flow, and 3 = complete mix. Machine learning wizards may bully me for this, but I am a mere machine learning novice."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.bayes[\"Digester Type\"].replace('Covered Lagoon',1, inplace = True)\n",
    "df.bayes[\"Digester Type\"].replace('Mixed Plug Flow',2, inplace =True)\n",
    "df.bayes[\"Digester Type\"].replace('Horizontal Plug Flow',2, inplace = True)\n",
    "df.bayes[\"Digester Type\"].replace('Vertical Plug Flow',2, inplace = True)\n",
    "df.bayes[\"Digester Type\"].replace('Plug Flow - Unspecified',2,inplace = True)\n",
    "df.bayes[\"Digester Type\"].replace('Complete Mix',3,inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Dairy</th>\n",
       "      <th>Biogas_gen_ft3_day</th>\n",
       "      <th>biogas/dairy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15500</td>\n",
       "      <td>600000</td>\n",
       "      <td>38.709677</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1700</td>\n",
       "      <td>50000</td>\n",
       "      <td>29.411765</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>9700</td>\n",
       "      <td>270000</td>\n",
       "      <td>27.835052</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>7000</td>\n",
       "      <td>360000</td>\n",
       "      <td>51.428571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>400</td>\n",
       "      <td>30000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108</th>\n",
       "      <td>3</td>\n",
       "      <td>10000</td>\n",
       "      <td>2739726</td>\n",
       "      <td>273.972600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>2</td>\n",
       "      <td>455</td>\n",
       "      <td>14000</td>\n",
       "      <td>30.769231</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>2</td>\n",
       "      <td>380</td>\n",
       "      <td>37500</td>\n",
       "      <td>98.684211</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>3</td>\n",
       "      <td>350</td>\n",
       "      <td>33450</td>\n",
       "      <td>95.571429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112</th>\n",
       "      <td>3</td>\n",
       "      <td>1240</td>\n",
       "      <td>2908000</td>\n",
       "      <td>2345.161290</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>113 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Digester Type  Dairy  Biogas_gen_ft3_day  biogas/dairy\n",
       "0                1  15500              600000     38.709677\n",
       "1                1   1700               50000     29.411765\n",
       "2                1   9700              270000     27.835052\n",
       "3                1   7000              360000     51.428571\n",
       "4                1    400               30000     75.000000\n",
       "..             ...    ...                 ...           ...\n",
       "108              3  10000             2739726    273.972600\n",
       "109              2    455               14000     30.769231\n",
       "110              2    380               37500     98.684211\n",
       "111              3    350               33450     95.571429\n",
       "112              3   1240             2908000   2345.161290\n",
       "\n",
       "[113 rows x 4 columns]"
      ]
     },
     "execution_count": 216,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.bayes['biogas/dairy'] = df.bayes['Biogas_gen_ft3_day']/df.bayes['Dairy']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are a couple outlandish biogas/dairy values. I filter the data so all data machine learned has biogas/dairy values within 2 standard deviations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 218,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pred_filter(data):\n",
    "    Y_upper = np.percentile(data['biogas/dairy'], 97.5)\n",
    "    Y_lower = np.percentile(data['biogas/dairy'], 2.5)\n",
    "    filtered_hist_data = data[(data['biogas/dairy'] >= Y_lower) & (data['biogas/dairy'] <= Y_upper)]\n",
    "    return filtered_hist_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Dairy</th>\n",
       "      <th>Biogas_gen_ft3_day</th>\n",
       "      <th>biogas/dairy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15500</td>\n",
       "      <td>600000</td>\n",
       "      <td>38.709677</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1700</td>\n",
       "      <td>50000</td>\n",
       "      <td>29.411765</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>9700</td>\n",
       "      <td>270000</td>\n",
       "      <td>27.835052</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>7000</td>\n",
       "      <td>360000</td>\n",
       "      <td>51.428571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>400</td>\n",
       "      <td>30000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>107</th>\n",
       "      <td>1</td>\n",
       "      <td>980</td>\n",
       "      <td>47000</td>\n",
       "      <td>47.959184</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108</th>\n",
       "      <td>3</td>\n",
       "      <td>10000</td>\n",
       "      <td>2739726</td>\n",
       "      <td>273.972600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>2</td>\n",
       "      <td>455</td>\n",
       "      <td>14000</td>\n",
       "      <td>30.769231</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>2</td>\n",
       "      <td>380</td>\n",
       "      <td>37500</td>\n",
       "      <td>98.684211</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>3</td>\n",
       "      <td>350</td>\n",
       "      <td>33450</td>\n",
       "      <td>95.571429</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>107 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Digester Type  Dairy  Biogas_gen_ft3_day  biogas/dairy\n",
       "0                1  15500              600000     38.709677\n",
       "1                1   1700               50000     29.411765\n",
       "2                1   9700              270000     27.835052\n",
       "3                1   7000              360000     51.428571\n",
       "4                1    400               30000     75.000000\n",
       "..             ...    ...                 ...           ...\n",
       "107              1    980               47000     47.959184\n",
       "108              3  10000             2739726    273.972600\n",
       "109              2    455               14000     30.769231\n",
       "110              2    380               37500     98.684211\n",
       "111              3    350               33450     95.571429\n",
       "\n",
       "[107 rows x 4 columns]"
      ]
     },
     "execution_count": 219,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred_filter(df.bayes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_bayes_clean = pred_filter(df.bayes).drop(columns=['biogas/dairy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Dairy</th>\n",
       "      <th>Biogas_gen_ft3_day</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15500</td>\n",
       "      <td>600000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1700</td>\n",
       "      <td>50000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>9700</td>\n",
       "      <td>270000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>7000</td>\n",
       "      <td>360000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>400</td>\n",
       "      <td>30000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>107</th>\n",
       "      <td>1</td>\n",
       "      <td>980</td>\n",
       "      <td>47000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108</th>\n",
       "      <td>3</td>\n",
       "      <td>10000</td>\n",
       "      <td>2739726</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>2</td>\n",
       "      <td>455</td>\n",
       "      <td>14000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>2</td>\n",
       "      <td>380</td>\n",
       "      <td>37500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>3</td>\n",
       "      <td>350</td>\n",
       "      <td>33450</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>107 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Digester Type  Dairy  Biogas_gen_ft3_day\n",
       "0                1  15500              600000\n",
       "1                1   1700               50000\n",
       "2                1   9700              270000\n",
       "3                1   7000              360000\n",
       "4                1    400               30000\n",
       "..             ...    ...                 ...\n",
       "107              1    980               47000\n",
       "108              3  10000             2739726\n",
       "109              2    455               14000\n",
       "110              2    380               37500\n",
       "111              3    350               33450\n",
       "\n",
       "[107 rows x 3 columns]"
      ]
     },
     "execution_count": 221,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_bayes_clean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "metadata": {},
   "outputs": [],
   "source": [
    "X1 = df_bayes_clean.drop([\"Biogas_gen_ft3_day\"], axis = 1)\n",
    "Y1 = df_bayes_clean[\"Biogas_gen_ft3_day\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, random_state = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train1 = np.array(x_train1)\n",
    "y_train1 = np.array(y_train1).squeeze()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 225,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nb.fit(x_train1, y_train1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:443: UserWarning: X has feature names, but MultinomialNB was fitted without feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "y_predicted = nb.predict(x_test1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.037037037037037035"
      ]
     },
     "execution_count": 227,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test1, y_predicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 228,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([132000])"
      ]
     },
     "execution_count": 228,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nb.predict([[2,500]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 229,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "11778.084345590833"
      ]
     },
     "execution_count": 229,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_squared_error(y_test1, y_predicted, squared = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 230,
   "metadata": {},
   "outputs": [],
   "source": [
    "def biogas_pred_nb(digester_type,dairy):\n",
    "    biogas = nb.predict([[digester_type,dairy]])\n",
    "    dairy_eff = nb.predict([[digester_type,dairy]])/dairy/101.336\n",
    "    return biogas, dairy_eff"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Predicting ft3 biogas/day (output 1) and digester efficiency (output 2) using Naive Bayes machine learning where input 1 is digester type (1 is impermeable cover, 2 is plug flow, 3 is complete mix) and input 2 is number of dairy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 290,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([1200000]), array([2.36835873]))"
      ]
     },
     "execution_count": 290,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "biogas_pred_nb(2,5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 291,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([1200000]), array([2.36835873]))"
      ]
     },
     "execution_count": 291,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "biogas_pred_nb(1,5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 292,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([432000]), array([0.85260914]))"
      ]
     },
     "execution_count": 292,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "biogas_pred_nb(3,5000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Random Forest\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Random forest machine learning is completed in this section to predict the daily biogas production from an inputted number of dairy cows and an inputted digester type. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_forest_clean = df_bayes_clean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 235,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Digester Type</th>\n",
       "      <th>Dairy</th>\n",
       "      <th>Biogas_gen_ft3_day</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15500</td>\n",
       "      <td>600000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1700</td>\n",
       "      <td>50000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>9700</td>\n",
       "      <td>270000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>7000</td>\n",
       "      <td>360000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>400</td>\n",
       "      <td>30000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>107</th>\n",
       "      <td>1</td>\n",
       "      <td>980</td>\n",
       "      <td>47000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108</th>\n",
       "      <td>3</td>\n",
       "      <td>10000</td>\n",
       "      <td>2739726</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>2</td>\n",
       "      <td>455</td>\n",
       "      <td>14000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>2</td>\n",
       "      <td>380</td>\n",
       "      <td>37500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>3</td>\n",
       "      <td>350</td>\n",
       "      <td>33450</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>107 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Digester Type  Dairy  Biogas_gen_ft3_day\n",
       "0                1  15500              600000\n",
       "1                1   1700               50000\n",
       "2                1   9700              270000\n",
       "3                1   7000              360000\n",
       "4                1    400               30000\n",
       "..             ...    ...                 ...\n",
       "107              1    980               47000\n",
       "108              3  10000             2739726\n",
       "109              2    455               14000\n",
       "110              2    380               37500\n",
       "111              3    350               33450\n",
       "\n",
       "[107 rows x 3 columns]"
      ]
     },
     "execution_count": 235,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_forest_clean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 236,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df_forest_clean.drop([\"Biogas_gen_ft3_day\"], axis = 1)\n",
    "y = df_forest_clean[\"Biogas_gen_ft3_day\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "import sklearn.ensemble as ske"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 238,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 239,
   "metadata": {},
   "outputs": [],
   "source": [
    "reg = ske.RandomForestRegressor(n_estimators = 1000, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 240,
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_train = np.ravel(Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 241,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(n_estimators=1000, random_state=0)"
      ]
     },
     "execution_count": 241,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 242,
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_pred = reg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 243,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import explained_variance_score,max_error, mean_absolute_error, mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 244,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "277429.45213811955"
      ]
     },
     "execution_count": 244,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_squared_error(Y_test, Y_pred, squared = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 245,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.1536456579754666"
      ]
     },
     "execution_count": 245,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "explained_variance_score(Y_test, Y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 246,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1253062.108"
      ]
     },
     "execution_count": 246,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max_error(Y_test, Y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 247,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([28353.144])"
      ]
     },
     "execution_count": 247,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.predict([[2,500]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 248,
   "metadata": {},
   "outputs": [],
   "source": [
    "def biogas_pred_rf(digester_type,dairy):\n",
    "    biogas = reg.predict([[digester_type,dairy]])\n",
    "    dairy_eff = reg.predict([[digester_type,dairy]])/dairy/101.336\n",
    "    return biogas, dairy_eff"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below are the random forest predictions for daily biogas production in ft3/day and efficiency (as a fraction, not percentage) for 5000 dairy cows for plug flow, then impermeable cover, then complete mixed anaerobic digester types. The random forest predictions are significantly better than the Naive Bayes predictions as noted by efficiency values that are less than 1."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 287,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names\n",
      "  warnings.warn(\n",
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([352650.592]), array([0.69600259]))"
      ]
     },
     "execution_count": 287,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "biogas_pred_rf(2,5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 288,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names\n",
      "  warnings.warn(\n",
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([270734.848]), array([0.53433103]))"
      ]
     },
     "execution_count": 288,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "biogas_pred_rf(1,5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 289,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names\n",
      "  warnings.warn(\n",
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([325077.044]), array([0.64158255]))"
      ]
     },
     "execution_count": 289,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "biogas_pred_rf(3,5000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Using the AgSTAR anaerobic digester database, this notebook calculates digester efficiencies for dairy manure (all types) digesters, dairy manure plug flow digesters, dairy manure complete mix digesters, and dairy manure impermeable cover digesters. This notebook also features linear regression analysis and 95% confidence intervals for all aforementioned digester types, where the independent variable is the number of dairy cows and the dependent variable is daily biogas production.\n",
    "\n",
    "Efficiencies of Digesters using 95% CI (2 standard deviation) are:\n",
    "\n",
    "Dairy- all anaerobic digester types: 68.8%\n",
    "\n",
    "Dairy plug flow: 73.7%\n",
    "\n",
    "Dairy complete mix: 85.9%\n",
    "\n",
    "Dairy impermeable cover: 43.1%\n",
    "\n",
    "Efficiencies of Digesters using 68% CI (1 standard deviation) are:\n",
    "\n",
    "Dairy- all anaerobic digester types: 65.8%\n",
    "\n",
    "Dairy plug flow: 71.3%\n",
    "\n",
    "Dairy complete mix: 82.7%\n",
    "\n",
    "Dairy impermeable cover: 41.1%\n",
    "\n",
    "Digester efficiencies for swine manure anaerobic digesters were not analyzed due to little and poor data.\n",
    "\n",
    "This notebook also features graphs relating impermeable cover digester daily biogas production to number of dairy or swine and location. Impermeable cover digesters are not heated, and therefore digestion efficiency should be heavily influenced by outdoor temperature. Based on the visuals of the graphs, there seems to be an insufficient amount of data and no correlation between greater biogas production per animal and southern states. This conclusion seems unlikely. With more data, we should see biogas production per animal influenced by the geographical location of the digester.\n",
    "\n",
    "Naive Bayes and Random Forest machine learning methods were used to predict daily biogas productions for dairy digesters given the number of dairy cows and type of digester. Daily biogas production should be approximately linearly related to the number of dairy cows, meaning the model should behave as a regression model. Naive Bayes machine learns uses categorization, and categorization is discrete, or not continuous like a regression model. Therefore, Naive Bayes is a poor machine learning method for this application. Random Forest machine learning can effectively model a continuous regression model. The random forest efficiency prediction are significantly closer the efficiency values calculated using the confidence intervals.\n",
    "\n",
    "Naive Bayes daily biogas and digester efficiency prediction for 5000 dairy cows:\n",
    "\n",
    "Plug flow: 1,200,000 ft3 biogas/day     236.8% efficient\n",
    "\n",
    "Complete mix: 432,000 ft3 biogas/day     85.3% efficient\n",
    "\n",
    "Impermeable cover: 1,200,000 ft3 biogas/day     236.6% efficient\n",
    "\n",
    "\n",
    "Random Forest daily biogas and digester efficiency prediction for 5000 dairy cows:\n",
    "\n",
    "Plug flow: 352,651 ft3 biogas/day     69.6% efficient\n",
    "\n",
    "Complete mix: 325,077 ft3 biogas/day     64.2% efficient\n",
    "\n",
    "Impermeable cover: 270,735 ft3 biogas/day   53.4% efficient\n",
    "\n",
    "Overall, digester efficiencies appear to be similar to that reported in literature. Nonetheless, more data would vastly improve anaerobic digestion efficiency analysis and the Random Forest model which predicts daily biogas production.\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
