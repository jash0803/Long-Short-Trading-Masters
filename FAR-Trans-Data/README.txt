# FAR-Trans: An Investment Dataset for Financial Asset Recommendation

Dataset description
--------------------
This dataset (FAR-Trans) is a dataset for financial asset recommendation, including basic information about financial securities (stocks, bonds and mutual funds), pricing time series for those assets, customer information and investment transactions provided by a large European financial institution.

Usage License
-------------
This dataset can be used under the CC-BY 4.0 license: https://creativecommons.org/licenses/by/4.0/.

Citation
--------
To acknowledge the use of this dataset in publications, please cite the following paper:
    
Javier Sanz-Cruzado, Nikolaos Droukas, Richard McCreadie. FAR-Trans: An Investment Dataset for Financial Asset Recommendation. IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys). Jeju, South Korea, August 2024.

Formatting and encoding
------------------------
A majority of the dataset files are written as comma-separated values (CSV) files with a single header row. Data points containing commas (,) are escaped using double quotes ("). In case any file is not in this format, it will be indicated in its description below. The enconding for all these files is UTF-8.

Files
-----
We incldue below the information available on each of the files, including file name, file description, and a description of the different columns of the file. For the column description, the name matches the one in the file, and the order on which the fields are introduced corresponds to the order in the file.

### Customer information

* File name: customer_information.csv

* Description: Detailed information about the customers. NOTE: A customer might appear here several times (with different dates, as their information might have been updated)

* Columns:
    - customerID: Customer identifier.
    - customerType: Type of the customer. It can have the following values:
        - Mass: Majority of customers. Less than 60k euros on investments.
        - Premium: Individual investor. More than 60k euros on investments.
        - Professional: Sole proprietorship. Individual customer that exercises his financial activity without having created a legal person.
        - Legal entity: legal entities with services within the bank.
        - Inactive: Customers that cannot be identified in the previous categories.
    - riskLevel: Risk tolerance of the customer. Levels, from lower to higher risk:
        - Conservative
        - Income
        - Balanced
        - Aggressive
        - Predicted_Conservative / Predicted_Income / Predicted_Balanced / Predicted_Aggressive: Same as earlier, but predicted by an algorithm.
        - Not_Available: Not available data.
    - investmentCapacity: Amount of money that the customer might invest. Possible values:
        - CAP_LT_30K: Less than 30K euros.
        - CAP_30K_80K: Between 30K and 80K euros.
        - CAP_80K_300K: Between 80K and 300K euros.
        - CAP_GT300K: More than 300K euros.
        - Predicted_CAP_LT_30K / Predicted_CAP_30K_80K / Predicted_CAP_80K_300K / Predicted_GT300K: Same as earlier, but predicted by an algorithm.
        - Not_Available: Not available data.
    - lastQuestionnaireDate: Date where the risk questionnaire was last completed.
- timestamp: Timestamp of last update.
    
### Asset information

* File name: asset_information.csv

* Description: Detailed information about the assets. NOTE: An asset can appear here several times.

* Columns:
    - ISIN: Unique International Securities Identification Number (ISIN) of the asset.
    - assetName: Name of the asset.
    - assetShortName: Short name of the asset.
    - assetCategory: Category of the asset. Values:
        - Stock
        - Bond
        - MTF (mutual fund)
    - assetSubCategory: Sub-category of the asset. Depends on the main type.
    - marketID: Market where the asset is traded.
    - sector: economic sector of the asset (when available).
    - industry: industry of the asset (when available).
    - timestamp: Last update date.

### Markets information

* File name: markets.csv

* Description: Basic information about the markets.

* Columns:
    - exchangeID: Stock exchange identifier.
    - marketID: Market identifier.
    - name: Name of the market.
    - description: Brief description of the market.
    - country: Country where the market is located.
    - tradingDays: Days of the week when the market is open (comma-separated list, possible values: Mon,Tue,Wed,Thu,Fri,Sat,Sun).
    - tradingHours: Hours of the day where the market is open (GMT). Format: (HH:MM-HH:MM).
    - marketClass: Type of market.

### Close prices

* File name: close_prices.csv

* Description: Contains the daily close prices for every asset in the dataset.

* Columns:
    - ISIN: Unique International Securities Identification Number (ISIN) of the asset.
    - timestamp: Timestamp (YYYY-mm-dd format).
    - closePrice: Close price of the asset (in euros).

### Limit prices

* File name: limit_prices.csv

* Description: Analysis of every asset in the dataset. Contains starting / end dates of time series for every asset, and values at the extremes.

* Columns:
    - ISIN: Unique International Securities Identification Number (ISIN) of the asset.
    - minDate: First date on which the asset has close price information.
    - maxDate: Last date on which the asset has close price information.
    - priceMinDate: Close price for the "minDate" date.
    - priceMaxDate: Close price for the "maxDate" date.
    - profitability: Return on investment (ROI) between "minDate" and "maxDate".

### Investment transactions

* File name: transactions.csv

* Description: Detailed investment transactions by customers.

* Columns:
    - customerID: Customer identifier.
    - ISIN: Unique International Securities Identification Number (ISIN) of the asset.
    - transactionID: Transaction identifier.
    - transactionType: Transaction type (Buy / Sell).
    - timestamp: Date of the transaction (YYYY-MM-DD format)
    - totalValue: Total monetary value of the transaction (in euros).
    - units: Number of shares traded.
    - channel: Channel over which the customer performed the transaction. Values:
        - Internet Banking
        - Phone Banking
        - Branch
    - marketID: Identifier of the market over which the transaction was performed.

### Questionnaires

* File name: questionnaires.csv

* Description: Questions for the risk MiFiD questionnaire (answers not provided).

* Format: Differently from other files, this is not a CSV formatted file, but, instead, a list of questions and possible answers for the questionnaire. The format for this file is:

    Section 1 Title

        1. Question 1 for Section 1
            a. Answer 1
            b. Answer 2
            ...
            n. Answer n

        2. Question 2 for Section 1
        ...

    Section 2 Title

        M+1. Question 1 for Section 2
        ...



