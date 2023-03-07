from django.shortcuts import render, redirect
from .models import Data,Employee, Otp
from django.http import HttpResponseRedirect
from django.http import HttpResponse
import logging
from django.core.mail import send_mail
from django.contrib.auth.decorators import login_required
from datetime import datetime
import pytz
from django.contrib import messages
import numpy as np

#data visulization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import sqlite3
import plotly.express as px
#########################################
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import pyotp

IST = pytz.timezone('Asia/Kolkata')

logging.basicConfig(filename='logger.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger()

def index(request):
    return render(request, 'login_page.html')
def api_page(request):
    if request.method == "POST":
        if request.POST.get('q34_employeeId') and request.POST.get('q37_domain') and request.POST.get('q35_howLong') and request.POST.get('q125_areYou') and request.POST.get('q103_typeA') and request.POST.get('q158_onA') and request.POST.get('q154_doYou154') and request.POST.get('q151_doYou') and request.POST.get('q104_onA104') and request.POST.get('q135_howSatisfied135') and request.POST.get('q147_howSatisfied147') and request.POST.get('q105_howSatisfied105') and request.POST.get('q137_nameThree') and request.POST.get('q106_howSatisfied106') and request.POST.get('q149_howSatisfied149'): #and request.post.get('q112_isThere') and request.post.get('q180_pleaseSpecify180') and request.post.get('q179_pleaseSpecify179') and request.post.get('q178_whatCan') and request.post.get('q177_whatExpect') and request.post.get('q170_typeHere') and request.post.get('q173_whatWill') and request.POST.get('q225_pleaseSpecify'): # and request.POST.get('q89_typeA89') :
            # logger.info(f"Data recived from employee {request.post.get('q34_employeeId')}")
            print("Data received")
            post = Data()
            post.Employee_ID = request.POST.get('q34_employeeId')
            post.Domain = request.POST.get('q37_domain')
            post.Other_domain = request.POST.get('q225_pleaseSpecify')
            post.Working_years = request.POST.get('q35_howLong')
            post.Bored = request.POST.get('q125_areYou')
            post.Free_time = request.POST.get('q173_whatWill')
            post.Satisfied_with_company = request.POST.get('q103_typeA')
            post.Improve_company = request.POST.get('q170_typeHere')
            post.Recommend_friends = request.POST.get('q158_onA')
            post.Working_team = request.POST.get('q154_doYou154')
            post.Team_improve = request.POST.get('q177_whatExpect')
            post.Coming_to_work = request.POST.get('q151_doYou')
            post.Satisfied_with_manager = request.POST.get('q104_onA104')
            post.Manager_improve = request.POST.get('q178_whatCan')
            post.Culture_Values = request.POST.get('q135_howSatisfied135')
            post.Compensation_Benefits = request.POST.get('q147_howSatisfied147')
            post.Satisfied_with_management = request.POST.get('q105_howSatisfied105')
            post.Management_improve = request.POST.get('q179_pleaseSpecify179')
            post.Improve_work = request.POST.get('q137_nameThree')
            post.Satisfied_with_HR = request.POST.get('q106_howSatisfied106')
            post.Hr_improve = request.POST.get('q180_pleaseSpecify180')
            post.Work_life_balance = request.POST.get('q149_howSatisfied149')
            post.Suggestions = request.POST.get('q112_isThere')

            sc=Data.objects.filter(Employee_ID=post.Employee_ID)
            if sc.count() > 0:
                Data.objects.filter(Employee_ID=post.Employee_ID).update(Domain=post.Domain, Other_domain=post.Other_domain ,Working_years=post.Working_years,Bored = post.Bored,Free_time=post.Free_time,Satisfied_with_company=post.Satisfied_with_company,Improve_company = post.Improve_company,Recommend_friends=post.Recommend_friends,Working_team = post.Working_team,Team_improve=post.Team_improve, Coming_to_work = post.Coming_to_work ,Satisfied_with_manager=post.Satisfied_with_manager,Manager_improve=post.Manager_improve,Culture_Values=post.Culture_Values,Compensation_Benefits= post.Compensation_Benefits,Satisfied_with_management=post.Satisfied_with_management,Management_improve=post.Management_improve,Improve_work = post.Improve_work,Satisfied_with_HR=post.Satisfied_with_HR,Hr_improve= post.Hr_improve,Work_life_balance=post.Work_life_balance,Suggestions=post.Suggestions)
                return HttpResponseRedirect('success/')
            else:
                post.save()
                print("Employee feedback saved successfully ")
                return HttpResponseRedirect('success/')
        else:
            print("IT is else part in success")
            return render(request, 'index.html')
    else:
        print("This is outer else part of success")
    return render(request, 'index.html')


def success(request):
    return render(request, 'messages.html')

def generate_secret_key():
    return pyotp.random_base32()

# Generate TOTP code
def generate_totp(secret_key):
    totp = pyotp.TOTP(secret_key)
    return totp.now()

def generate_mail(Employee_Mail):
    Db_validate = Otp()
    secret_key = generate_secret_key()
    totp = generate_totp(secret_key)
    Db_validate.Employee_Secret_key = secret_key
    Db_validate.Otp = totp
    Db_validate.Time = datetime.now()
    Db_validate.Employee_Mail = Employee_Mail
    print("Employee_Mail:", Employee_Mail)
    print("Generated_secretkey:", secret_key)
    print("Generated_Otp:", totp)
    send_mail(subject='OTP Validation',
              message="Hi \nPleae use the following otp to login \n Here is your OTP: {} \nThis Otp will expire in 5 minutes".format(
                  totp), from_email='pythonrgt@gmail.com', recipient_list=[Employee_Mail], fail_silently=False)
    sc = Otp.objects.filter(Employee_Mail=Db_validate.Employee_Mail)
    if sc.count() > 0:
        Otp.objects.filter(Employee_Mail=Db_validate.Employee_Mail).update(
            Employee_Secret_key=Db_validate.Employee_Secret_key, Time=Db_validate.Time, Otp=Db_validate.Otp)
    else:
        Db_validate.save()



def send_otp(request):
    if request.method == "POST":
        if request.POST.get('Employee_Mail'):
            Db_validate = Otp()
            global Employee_Mail
            Employee_Mail = request.POST.get('Employee_Mail')
            generate_mail(Employee_Mail)
            return render(request, 'otp_validate.html')
    else:
        return render(request, 'login_page.html')

def resent_otp(request):
    print("Employee_mail:",Employee_Mail)
    generate_mail(Employee_Mail)
def validate_otp(request):
    if request.method == "POST":
        Employee_Mail = request.POST.get('Employee_Mail')
        Employee_Otp = request.POST.get('otp')
        print("Entred_Mail:",Employee_Mail)
        print("Entred_Otp:",Employee_Otp)
        otp_verified = False

        # Get the TOTP secret key for the user
        otp_entry = Otp.objects.filter(Employee_Mail=Employee_Mail).first()
        sec_key = otp_entry.Employee_Secret_key
        db_otp = otp_entry.Otp
        print("Database_Otp:",db_otp)
        # if otp_entry:
        #     # Generate TOTP for the secret key
        #     totp = pyotp.TOTP(sec_key)
        #     # Verify the OTP entered by the user against the generated TOTP
        #     otp_verified = totp.verify(Employee_Otp)
        #     print("otp_verified:",otp_verified)
  
        if db_otp==Employee_Otp:
            # Check if the TOTP code is entered within 5 minutes
            Db_time = Otp.objects.filter(Employee_Mail=Employee_Mail).values_list('Time', flat=True)
            for_time = list(Db_time)
            t1 = for_time[0]
            t2 = t1.replace(tzinfo=None)
            time_difference = datetime.now() - t2
            if time_difference.total_seconds() <= 300:
                error_message = "OTP is validated, Redirect to Feedback."
                return render(request, 'index.html',{'error_message':error_message})
            else:
                otp_verified = False
                messages.error(request, 'OTP expired. Please try again.')
        else:
            messages.error(request, 'Invalid OTP. Please try again.')
    error_message = "OTP is Incorrect, Please try again."
    return render(request, 'otp_validate.html', {'error_message': error_message})

def results(request):
    con = sqlite3.connect('db.sqlite3')
    df = pd.read_sql_query("SELECT * from employee_feedback_data", con)

    df['Satisfied_with_company'] = df['Satisfied_with_company'].astype(str).astype(int)
    df['Compensation_Benefits'] = df['Compensation_Benefits'].astype(str).astype(int)
    df['Work_life_balance'] = df['Work_life_balance'].astype(str).astype(int)
    df['Culture_Values'] = df['Culture_Values'].astype(str).astype(int)
    df['Satisfied_with_HR'] = df['Satisfied_with_HR'].astype(str).astype(int)
    df['Satisfied_with_manager'] = df['Satisfied_with_manager'].astype(str).astype(int)
    df['Satisfied_with_management'] = df['Satisfied_with_management'].astype(str).astype(int)

    group1_count = df[(df['Satisfied_with_company'] >= 1) & (df['Satisfied_with_company'] <= 3)]['Satisfied_with_company'].count()
    group2_count = df[(df['Satisfied_with_company'] >= 4) & (df['Satisfied_with_company'] <= 7)]['Satisfied_with_company'].count()
    group3_count = df[(df['Satisfied_with_company'] >= 8) & (df['Satisfied_with_company'] <= 10)]['Satisfied_with_company'].count()
    group1_df = pd.DataFrame({'rating group': '1-3', 'count': group1_count}, index=[0])
    group2_df = pd.DataFrame({'rating group': '4-7', 'count': group2_count}, index=[0])
    group3_df = pd.DataFrame({'rating group': '8-10', 'count': group3_count}, index=[0])
    df1 = pd.DataFrame({'rating group': ['1-3', '4-7', '8-10'], 'count': [group1_count, group2_count, group3_count]})

    # Create a new figure object for the pie chart
    fig1, ax1 = plt.subplots()

    # Plot the pie chart
    ax1.pie(df1['count'], labels=df1['rating group'], autopct='%1.1f%%')
    ax1.set_title('Satisfied with company')

    # Save the pie chart as an image
    fig1.savefig('employee_feedback/static/graphs/pie_company.png')

    ###########################################################################################
    Domain = df['Domain']
    management = df['Satisfied_with_management']

    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(Domain, management)

    # Add x, y gridlines
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

    # Add Plot Title
    ax.set_title('Satisfied with Management',
                 loc='right', )
    plt.savefig(r'employee_feedback\static\graphs\bar_management.png')

    ######################################################################################3
    fig4, ax4 = plt.subplots(figsize=(15, 5))
    df.plot(x="Domain", y="Satisfied_with_manager",ax=ax4)
    plt.xlabel("Domain")
    plt.ylabel("Satisfied with Manager")
    plt.title("Satisfaction with Manager by Domain")
    fig4.savefig('employee_feedback/static/graphs/manager.png')


#################################################################################################
    # ratings = df["Satisfied_with_company"].value_counts()
    # numbers = ratings.index
    # quantity = ratings.values
    #
    # figure = px.pie(df,values=quantity,names=numbers, hole=0.5, title='Overall percentage Satisfied with company')
    # ratings1 = df['Satisfied_with_management'].value_counts()
    # numbers1 = ratings1.index
    # quantity1 = ratings1.values
    # figure1 = px.pie(df, values=quantity1, names=numbers1, hole=0.5, title='Overall percentage Satisfied with management')
    # figure.write_image('employee_feedback/static/graphs/Overall_comapany.jpeg')
    # figure1.write_image('employee_feedback/static/graphs/Overall_management.jpeg')

    fig2, ax2 = plt.subplots()
    # pass the axes object to plot function
    df.plot(kind='line', x='Domain', y='Compensation_Benefits', ax=ax2);
    df.plot(kind='line', x='Domain', y='Culture_Values', ax=ax2);
    df.plot(kind='line', x='Domain', y='Work_life_balance', ax=ax2);
    ax2.set_title('Employee Feedback');
    ax2.legend(['Compensation_Benefits', 'Culture_Values', 'Work_life_balance'])
    fig2.savefig('employee_feedback/static/graphs/plot.png')


    ###############################################################################
    fig3, ax3 = plt.subplots(figsize=(15, 6))

    sns.barplot(x='Domain', y='Satisfied_with_HR', data=df,ax=ax3)
    ax3.set_title('Satisfied With HR');
    fig3.savefig('employee_feedback/static/graphs/HR.png',dpi=100)


    return render(request,'graphs.html')


# def results(request):
#     con = sqlite3.connect('db.sqlite3')
#
#     data = pd.read_sql_query("SELECT * FROM employee_feedback_data WHERE Employee_ID = 'admin@ratnaglobaltech.com'",
#                              con)
#     # data.drop(['Unnamed: 0'], axis=1, inplace=True)
#     data.drop(['Employee_ID'], axis=1, inplace=True)
#     data.drop(['Domain'], axis=1, inplace=True)
#     data.drop(['Team_improve'], axis=1, inplace=True)
#     data.drop(['Coming_to_work'], axis=1, inplace=True)
#     data.drop(['Manager_improve'], axis=1, inplace=True)
#     data.drop(['Management_improve'], axis=1, inplace=True)
#     data.drop(['Improve_work'], axis=1, inplace=True)
#     data.drop(['Hr_improve'], axis=1, inplace=True)
#     data.drop(['Suggestions'], axis=1, inplace=True)
#     data.drop(['Improve_company'], axis=1, inplace=True)
#     data.drop(['Other_domain'], axis=1, inplace=True)
#     data.drop(['Free_time'], axis=1, inplace=True)
#
#     labelencoder = LabelEncoder()
#     data['Bored'] = labelencoder.fit_transform(data['Bored'])
#     data['Recommend_friends'] = labelencoder.fit_transform(data['Recommend_friends'])
#     data['Working_team'] = labelencoder.fit_transform(data['Working_team'])
#     data['Working_years'] = labelencoder.fit_transform(data['Working_years'])
#     data['Work_life_balance'] = labelencoder.fit_transform(data['Work_life_balance'])
#
#     X = data.drop('Bored', axis=1)
#     y = data['Satisfied_with_company']
#
#     model = LinearRegression()
#     model.fit(X, y)
#
#     y_pred = model.predict(X)
#     mse = mean_squared_error(y, y_pred)
#     print("Mean Squared Error: ", mse)
#
#     context = {
#         'mse':mse
#     }
#     return HttpResponse(request, "<h1>Hello {{ mse }}, how are you?</h1>")


def logout(request):
    return render(request, 'login_page.html')



# from django.contrib.auth import authenticate, login
# from django.shortcuts import render, redirect
# from .forms import TOTPLoginForm
#
# def login_view(request):
#     if request.method == 'POST':
#         form = TOTPLoginForm(request.POST)
#         if form.is_valid():
#             # Authenticate the user
#             username = form.cleaned_data['username']
#             password = form.cleaned_data['password']
#             user = authenticate(request, username=username, password=password)
#
#             # Check if the user is authenticated
#             if user is not None:
#                 # Get the user's TOTP secret key
#                 user_profile = user.userprofile
#                 totp_secret_key = user_profile.totp_secret_key
#
#                 # Verify the TOTP code
#                 totp_code = form.cleaned_data['totp_code']
#                 is_valid = pyotp.TOTP(totp_secret_key).verify(totp_code)
#
#                 if is_valid:
#                     # Log in the user
#                     login(request, user)
#                     return redirect('home')
#                 else:
#                     # Display an error message
#                     form.add_error('totp_code', 'Invalid TOTP code')
#     else:
#         form = TOTPLoginForm()
#     return render(request, 'login_page.html', {'form': form})
#
def model(request):
    con = sqlite3.connect('db.sqlite3')
    cmt = pd.read_sql_query("SELECT * FROM employee_feedback_data",con)
    cmt.drop(['id'], axis=1, inplace=True)
    cmt.drop(['Employee_ID'], axis=1, inplace=True)
    cmt.drop(['Team_improve'], axis=1, inplace=True)
    cmt.drop(['Coming_to_work'], axis=1, inplace=True)
    cmt.drop(['Manager_improve'], axis=1, inplace=True)
    cmt.drop(['Management_improve'], axis=1, inplace=True)
    cmt.drop(['Improve_work'], axis=1, inplace=True)
    cmt.drop(['Hr_improve'], axis=1, inplace=True)
    cmt.drop(['Suggestions'], axis=1, inplace=True)
    cmt.drop(['Improve_company'], axis=1, inplace=True)
    cmt.drop(['Other_domain'], axis=1, inplace=True)
    cmt.drop(['Free_time'], axis=1, inplace=True)
    from sklearn.preprocessing import LabelEncoder
    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    cmt['Bored'] = labelencoder.fit_transform(cmt['Bored'])
    cmt['Recommend_friends'] = labelencoder.fit_transform(cmt['Recommend_friends'])
    cmt['Working_team'] = labelencoder.fit_transform(cmt['Working_team'])
    cmt['Working_years'] = labelencoder.fit_transform(cmt['Working_years'])
    cmt['Work_life_balance'] = labelencoder.fit_transform(cmt['Work_life_balance'])

    cmt['Satisfied_with_company'] = cmt['Satisfied_with_company'].astype(str).astype(int)
    cmt['Compensation_Benefits'] = cmt['Compensation_Benefits'].astype(str).astype(int)
    cmt['Work_life_balance'] = cmt['Work_life_balance'].astype(str).astype(int)
    cmt['Culture_Values'] = cmt['Culture_Values'].astype(str).astype(int)
    cmt['Satisfied_with_HR'] = cmt['Satisfied_with_HR'].astype(str).astype(int)
    cmt['Satisfied_with_manager'] = cmt['Satisfied_with_manager'].astype(str).astype(int)
    cmt['Satisfied_with_management'] = cmt['Satisfied_with_management'].astype(str).astype(int)
    # Exploratory data analysis:
    # 1. Measures of central tendency
    # 2. Measures of dispersion
    # 3. Third moment business decision
    # 4. Fourth moment business decision
    # 5. Probability distributions of variables
    # 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
    cmt.mean()
    cmt.median()
    cmt.var()
    cmt.std()
    cmt.skew()
    cmt.kurt()
    cmt.isna().sum()  # information about missing values
    cmt.describe()

    #################################Bar plot, hist, boxplot #########################
    ############################ Univariate analysis ##########################
    ############################ Outlier treatment ############################
    # Graphical Representation
    import matplotlib.pyplot as plt  # mostly used for visualization purposes

    # Satisfied_with_company
    # plt.hist(cmt.Satisfied_with_company)  # histogram

    # plt.boxplot(cmt.Satisfied_with_company)  # boxplot

    import seaborn as sns

    # sns.boxplot(cmt['Satisfied_with_company'])

    IQR = cmt['Satisfied_with_company'].quantile(0.75) - cmt['Satisfied_with_company'].quantile(0.25)
    lower_limit = cmt['Satisfied_with_company'].quantile(0.25) - (IQR * 1.5)
    upper_limit = cmt['Satisfied_with_company'].quantile(0.75) + (IQR * 1.5)
    outliers_df = np.where(cmt['Satisfied_with_company'] > upper_limit, True,
                           np.where(cmt['Satisfied_with_company'] < lower_limit, True, False))
    cmt_trimmed = cmt.loc[~(outliers_df),]
    cmt.shape, cmt_trimmed.shape

    # let's explore outliers in the trimmed dataset
    # sns.boxplot(cmt_trimmed.Satisfied_with_company);
    # plt.title('Boxplot');
    # plt.show()

    ############################### Bivariate analysis #######################
    # Jointplot
    # sns.jointplot(x=cmt['Satisfied_with_management'], y=cmt['Satisfied_with_company'])

    # Countplot
    # plt.figure(1, figsize=(16, 10))
    # sns.countplot(cmt['Satisfied_with_management'])
    #
    # plt.figure(1, figsize=(16, 10))
    # sns.countplot(cmt['Bored'])
    #
    # plt.figure(1, figsize=(16, 10))
    # sns.countplot(cmt['Recommend_friends'])
    #
    # plt.figure(1, figsize=(16, 10))
    # sns.countplot(cmt['Culture_Values'])
    #
    # plt.figure(1, figsize=(16, 10))
    # sns.countplot(cmt['Compensation_Benefits'])
    #
    # plt.figure(1, figsize=(16, 10))
    # sns.countplot(cmt['Satisfied_with_HR'])
    #
    # plt.figure(1, figsize=(16, 10))
    # sns.countplot(cmt['Work_life_balance'])

    print("**********************************************************")

    # Q-Q Plot
    from scipy import stats
    import pylab
    stats.probplot(cmt.Satisfied_with_company, dist="norm", plot=pylab)
    # plt.show()

    # Scatter plot between the variables along with histograms
    import seaborn as sns
    # sns.pairplot(cmt.iloc[:, :])

    # Correlation matrix
    cmt.corr()

    ############################## Model Building #######################
    # we see there exists High collinearity between input variables especially between

    # preparing model considering all the variables
    import statsmodels.formula.api as smf  # for regression model

    ml1 = smf.ols(
        'Satisfied_with_company ~ Satisfied_with_manager + Culture_Values + Compensation_Benefits + Satisfied_with_management + Satisfied_with_HR + Work_life_balance',
        data=cmt).fit()  # regression model

    # Summary
    ml1.summary()

    # Checking whether data has any influential values
    # Influence Index Plots
    import statsmodels.api as sm

    # sm.graphics.influence_plot(ml1)

    # Studentized Residuals = Residual/standard deviation of residuals
    # index 1700,1440 is showing high influence so we can exclude that entire row

    cmt_new = cmt.drop(cmt.index[[23]])
    cmt_new = cmt.drop(cmt.index[[1]])

    # Preparing model
    ml_new = smf.ols(
        'Satisfied_with_company ~ Satisfied_with_manager + Culture_Values + Compensation_Benefits + Satisfied_with_management + Satisfied_with_HR + Work_life_balance',
        data=cmt_new).fit()

    # Summary
    ml_new.summary()

    # Check for Colinearity to decide to remove a variable using VIF
    # Assumption: VIF > 10 = colinearity
    # calculating VIF's values of independent variables
    rsq_Satisfied_with_manager = smf.ols(
        'Satisfied_with_manager ~ Culture_Values + Compensation_Benefits + Satisfied_with_management + Satisfied_with_HR + Work_life_balance',
        data=cmt).fit().rsquared
    vif_Satisfied_with_manager = 1 / (1 - rsq_Satisfied_with_manager)

    rsq_Culture_Values = smf.ols(
        'Culture_Values ~ Satisfied_with_manager + Compensation_Benefits + Satisfied_with_management + Satisfied_with_HR + Work_life_balance',
        data=cmt).fit().rsquared
    vif_Culture_Values = 1 / (1 - rsq_Culture_Values)

    rsq_Compensation_Benefits = smf.ols(
        'Compensation_Benefits ~ Culture_Values + Satisfied_with_manager + Satisfied_with_management + Satisfied_with_HR + Work_life_balance',
        data=cmt).fit().rsquared
    vif_Compensation_Benefits = 1 / (1 - rsq_Compensation_Benefits)

    rsq_Satisfied_with_management = smf.ols(
        'Satisfied_with_management ~ Culture_Values + Satisfied_with_manager + Compensation_Benefits + Satisfied_with_HR + Work_life_balance',
        data=cmt).fit().rsquared
    vif_Satisfied_with_management = 1 / (1 - rsq_Satisfied_with_management)

    rsq_Satisfied_with_HR = smf.ols(
        'Satisfied_with_HR ~ Satisfied_with_management + Culture_Values + Satisfied_with_manager + Compensation_Benefits + Work_life_balance',
        data=cmt).fit().rsquared
    vif_Satisfied_with_HR = 1 / (1 - rsq_Satisfied_with_HR)

    rsq_Work_life_balance = smf.ols(
        'Work_life_balance ~ Satisfied_with_HR + Satisfied_with_management + Culture_Values + Satisfied_with_manager + Compensation_Benefits',
        data=cmt).fit().rsquared
    vif_Work_life_balance = 1 / (1 - rsq_Work_life_balance)

    # Storing vif values in a data frame
    d1 = {
        'Variables': ['Satisfied_with_manager', 'Culture_Values', 'Compensation_Benefits', 'Satisfied_with_management',
                      'Satisfied_with_HR', 'Work_life_balance'],
        'VIF': [vif_Satisfied_with_manager, vif_Culture_Values, vif_Compensation_Benefits,
                vif_Satisfied_with_management, vif_Satisfied_with_HR, vif_Work_life_balance]}
    Vif_frame = pd.DataFrame(d1)
    Vif_frame
    # As Satisfied_with_management is having highest VIF value, we are going to drop this from the prediction model
    # As Work_life_balance is having lowest VIF value, we are going to improve in this area.

    cmt.drop(['Satisfied_with_management'], axis=1, inplace=True)

    # Final model
    final_ml = smf.ols(
        'Satisfied_with_company ~ Satisfied_with_manager + Culture_Values + Compensation_Benefits + Satisfied_with_HR + Work_life_balance',
        data=cmt).fit()
    final_ml.summary()

    # Prediction
    pred = final_ml.predict(cmt)

    # Q-Q plot
    res = final_ml.resid
    sm.qqplot(res)
    # plt.show()

    # Q-Q plot
    stats.probplot(res, dist="norm", plot=pylab)
    # plt.show()

    # Residuals vs Fitted plot
    sns.residplot(x=pred, y=cmt.Satisfied_with_company, lowess=True)
    plt.xlabel('Fitted')
    plt.ylabel('Residual')
    plt.title('Fitted vs Residual')
    # plt.show()

    sm.graphics.influence_plot(final_ml)

    ### Splitting the data into train and test data
    from sklearn.model_selection import train_test_split
    cmt_train, cmt_test = train_test_split(cmt, test_size=0.2)  # 20% test data

    # preparing the model on train data
    model_train = smf.ols(
        "Satisfied_with_company ~ Satisfied_with_manager + Culture_Values + Compensation_Benefits + Satisfied_with_HR + Work_life_balance",
        data=cmt_train).fit()

    # prediction on test data set
    test_pred = model_train.predict(cmt_test)

    # test residual values
    test_resid = test_pred - cmt_test.Satisfied_with_company
    # RMSE value for test data
    test_rmse = np.sqrt(np.mean(test_resid * test_resid))
    test_rmse

    # train_data prediction
    train_pred = model_train.predict(cmt_train)

    # train residual values
    train_resid = train_pred - cmt_train.Satisfied_with_company
    # RMSE value for train data
    train_rmse = np.sqrt(np.mean(train_resid * train_resid))
    train_rmse

    """Model is right fit"""

    Vif_frame.sort_values(by=["VIF"])

    d1 = Vif_frame.sort_values(by=["VIF"])

    d2 = d1.head(2)

    print(d2)

    d3 = d2.Variables
    d3 = d2.astype({'Variables':'string'})
    d3['Variables'] = d3['Variables']
    print(d3.dtypes)
    print(d3)
    print(d3.to_string(header=False))

    prefix = "Thank you for your feedback: "
    suffix = "fields are having the highest disatisfaction factor. Management will try to improve in this area."
    variables = d3.loc[:, "Variables"].to_list()

    output = f"{prefix}{' & '.join(variables)} {suffix}"
    print(output)
    # context = {'d2':d2}
    # df_html = d1.to_html()

    # pass the HTML string as context to the template
    context = {'df_html': output}
    return render(request, 'model.html', context)
    # return render(request, 'model.html',context=context)