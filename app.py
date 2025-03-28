import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import shap
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from PIL import Image
import base64
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Football Mistake Dashboard")
st.title("📊 Football Mistake Analytics Dashboard")
# Load data once
@st.cache_data
def load_data():
    return pd.read_excel('new_data.xlsx', sheet_name='Export')

def load_local_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

df = load_data()
logo = load_local_image_as_base64("logo.png")

# ------------------------ KPI BAR --------------------------
def show_kpis(df):
    total_mistakes = df.shape[0]
    unique_statisticians = df['Statistician (Adjusted) Name'].nunique()
    total_competitions = df['Competition'].nunique()

    st.markdown(
        """
        <style>
        .kpi-card {
            background-color: #D3D3D3;
            padding: 1.2rem;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: bold;
            color: #0e1117;
        }
        .kpi-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{total_mistakes}</div>
                <div class="kpi-label">Total Mistakes</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{unique_statisticians}</div>
                <div class="kpi-label">Statisticians Involved</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{total_competitions}</div>
                <div class="kpi-label">Competitions</div>
            </div>
        """, unsafe_allow_html=True)


# ------------------------ PLOT FUNCTIONS --------------------------

def plot_annual_mistake_trend():
    df1 = df.copy()
    df1['StartDate'] = pd.to_datetime(df1['StartDate'], errors='coerce')
    df1 = df1.dropna(subset=['StartDate'])

    df1['Year'] = df1['StartDate'].dt.year

    # Overall yearly mistakes
    overall = df1.groupby('Year').size().reset_index(name='MistakeCount')

    # England only
    england = df1[df1['Country'] == 'England'].groupby('Year').size().reset_index(name='MistakeCount')

    # Scotland only
    scotland = df1[df1['Country'] == 'Scotland'].groupby('Year').size().reset_index(name='MistakeCount')

    fig = go.Figure()

    # Overall line
    fig.add_trace(go.Scatter(
        x=overall['Year'],
        y=overall['MistakeCount'],
        mode='lines+markers',
        name='Overall',
        line=dict(color='darkgreen', width=5),
        marker=dict(
            color='darkgreen',
            size=12,
            line=dict(width=3, color='lightgreen'),
            opacity=1,
            symbol='circle'
        )
    ))

    # England trend
    fig.add_trace(go.Scatter(
        x=england['Year'],
        y=england['MistakeCount'],
        mode='lines+markers',
        name='England',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(
            color='red',
            size=10,
            line=dict(width=3, color='tomato'),
            opacity=1,
            symbol='circle'
        )
    ))

    # Scotland trend
    fig.add_trace(go.Scatter(
        x=scotland['Year'],
        y=scotland['MistakeCount'],
        mode='lines+markers',
        name='Scotland',
        line=dict(color='blue', width=2, dash='dot'),
        marker=dict(
            color='blue',
            size=10,
            line=dict(width=3, color='skyblue'),
            opacity=1,
            symbol='circle'
        )
    ))

    fig.update_layout(
        title='Annual Mistake Trends (Overall vs England vs Scotland)',
        xaxis_title='Year',
        yaxis_title='Number of Mistakes',
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        hovermode='x unified'
    )

    return fig

def plot_yearly_mistake_type_distribution():
    df1 = df.copy()
    df1['StartDate'] = pd.to_datetime(df1['StartDate'], errors='coerce')
    df1['Year'] = df1['StartDate'].dt.year.astype(str)

    # Group by Year and MistakeTypeCategory
    mistake_type_yearly = df1.groupby(["Year", "MistakeTypeCategory"]).size().reset_index(name="Count")

    # Define 11 red shades (light to dark)
    red_shades = [
        '#FFE5E5', '#FFCCCC', '#FF9999', '#FF6666', '#FF4D4D',
        '#FF3333', '#FF1A1A', '#FF0000', '#CC0000', '#990000', '#660000'
    ]
    categories = mistake_type_yearly["MistakeTypeCategory"].unique()
    shade_map = {cat: red_shades[i % len(red_shades)] for i, cat in enumerate(categories)}

    # Create bar chart
    fig = px.bar(
        mistake_type_yearly,
        x="Year",
        y="Count",
        color="MistakeTypeCategory",
        title="Yearly Distribution of MistakeTypeCategory",
        barmode="group",
        template="plotly_dark",
        color_discrete_map=shade_map,
        opacity=0.7
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Mistake Count",
        legend_title="Mistake Type Category",
        font=dict(color='white'),
        plot_bgcolor='black',
        paper_bgcolor='black'
    )

    return fig

def plot_mistake_type_distribution():
    df2 = df.dropna(subset=['MistakeType'])
    mistake_type_counts = df2['MistakeType'].value_counts().reset_index()
    mistake_type_counts.columns = ['MistakeType', 'Count']
    fig = px.pie(mistake_type_counts, names='MistakeType', values='Count', hole=0.45,
                 title='Distribution of Mistake Types',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_traces(textinfo='percent+label', pull=[0.05]*len(mistake_type_counts),
                      marker=dict(line=dict(color='black', width=1.5)))
    fig.update_layout(title_font_size=22, plot_bgcolor='white', paper_bgcolor='black',
                      font=dict(color='black'))
    return fig

def plot_delay_duration_frequency():
    df3 = df[df['CategoryOption'].notna()]
    df3 = df3[df3['CategoryOption'].str.contains("Delayed", case=False, na=False)]
    delay_counts = df3['CategoryOption'].value_counts().reset_index()
    delay_counts.columns = ['DelayCategory', 'Count']
    fig = px.bar(delay_counts, x='Count', y='DelayCategory', orientation='h',
                 title='Delay Duration Frequency Analysis', color='Count',
                 color_continuous_scale='Plasma', template='plotly_white')
    fig.update_layout(title_font_size=22, xaxis_title='Number of Occurrences',
                      yaxis_title='Delay Category', plot_bgcolor='black', paper_bgcolor='black',
                      font=dict(color='black'), yaxis=dict(categoryorder='total ascending'))
    return fig

def plot_bubble_chart():
    event_summary = df.groupby('Event').agg({
        'Mistake ID': 'count',
        'Yellow ': 'sum',
        'Corner': 'sum'
    }).reset_index()
    event_summary.columns = ['Event', 'TotalMistakes', 'TotalYellows', 'TotalCorners']
    event_summary = event_summary.dropna(subset=['TotalYellows', 'TotalCorners'], how='all')
    fig = px.scatter(event_summary, x='TotalYellows', y='TotalMistakes', size='TotalCorners',
                     color='TotalCorners', hover_name='Event',
                     title='Mistakes vs Yellow Cards (Bubble Size = Corners)',
                     labels={'TotalYellows': 'Yellow Cards',
                             'TotalMistakes': 'Number of Mistakes',
                             'TotalCorners': 'Corner Kicks'},
                     template='plotly_dark', color_continuous_scale='Viridis')
    fig.update_layout(title_font_size=22, plot_bgcolor='black', paper_bgcolor='black',
                      font=dict(color='white'), coloraxis_colorbar=dict(title='Corners'))
    return fig

def plot_top_competitions():
    competition_mistakes = df['Competition'].value_counts().reset_index()
    competition_mistakes.columns = ['Competition', 'MistakeCount']
    top_competitions = competition_mistakes.head(10)
    fig = px.bar(top_competitions, x='Competition', y='MistakeCount',
                 title='Top Competitions by Number of Mistakes',
                 color='MistakeCount', color_continuous_scale='Tealgrn',
                 template='plotly_white')
    fig.update_traces(marker_line_color='black', marker_line_width=1.2)
    fig.update_layout(title_font_size=22, xaxis_title='Competition',
                      yaxis_title='Number of Mistakes', xaxis_tickangle=45,
                      plot_bgcolor='black', paper_bgcolor='black',
                      font=dict(color='black'), coloraxis_showscale=False)
    return fig

def plot_top_statisticians():
    statistician_mistakes = df['Statistician (Adjusted) Name'].value_counts().reset_index()
    statistician_mistakes.columns = ['Statistician', 'MistakeCount']
    top_statisticians = statistician_mistakes.head(15)
    fig = px.bar(top_statisticians, x='MistakeCount', y='Statistician',
                 orientation='h', title='Top Statisticians by Number of Mistakes',
                 color='MistakeCount', color_continuous_scale='Bluered_r',
                 template='plotly_dark')
    fig.update_traces(marker_line_color='black', marker_line_width=1.2)
    fig.update_layout(title_font_size=22, xaxis_title='Number of Mistakes',
                      yaxis_title='Statistician', plot_bgcolor='black',
                      paper_bgcolor='black', font=dict(color='white'),
                      yaxis=dict(categoryorder='total ascending'),
                      coloraxis_showscale=False)
    return fig

def plot_monthly_mistake_trend():
    df_time = df.copy()

    # Ensure StartDate is in datetime format
    df_time['StartDate'] = pd.to_datetime(df_time['StartDate'], errors='coerce')
    df_time = df_time.dropna(subset=['StartDate'])

    # Create Month-Year column
    df_time['MonthYear'] = df_time['StartDate'].dt.to_period('M').astype(str)

    # Group by Month-Year and count mistakes
    monthly_mistakes = df_time.groupby('MonthYear').size().reset_index(name='MistakeCount')

    # Convert back to datetime for plotting
    monthly_mistakes['MonthYear'] = pd.to_datetime(monthly_mistakes['MonthYear'])

    # Plot
    fig = px.line(
        monthly_mistakes,
        x='MonthYear',
        y='MistakeCount',
        title='Monthly Trend of Mistakes Across All Competitions',
        markers=True,
        template='plotly_dark',
        line_shape='spline'
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    fig.update_layout(
        title_font_size=22,
        xaxis_title='Month-Year',
        yaxis_title='Number of Mistakes',
        font=dict(color='white'),
        plot_bgcolor='black',
        paper_bgcolor='black',
        hovermode='x unified'
    )

    return fig


def plot_geographic_trend():
    df7 = df[df['Country'].isin(['England', 'Scotland'])].copy()
    df7['StartDate'] = pd.to_datetime(df7['StartDate'], errors='coerce')
    df7['MonthYear'] = df7['StartDate'].dt.to_period('M').astype(str)
    mistakes_by_country = df7.groupby(['MonthYear', 'Country']).size().reset_index(name='MistakeCount')
    mistakes_by_country['MonthYear'] = pd.to_datetime(mistakes_by_country['MonthYear'])
    fig = px.line(mistakes_by_country, x='MonthYear', y='MistakeCount', color='Country',
                  title='Geographic Impact: Mistakes Over Time (England vs Scotland)',
                  markers=True, template='plotly_dark')
    fig.add_shape(type="line", x0="2023-10-01", x1="2023-10-01",
                  y0=0, y1=mistakes_by_country['MistakeCount'].max(),
                  line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x="2023-10-01", y=mistakes_by_country['MistakeCount'].max(),
                       text="VAR Introduced in Scotland", showarrow=True,
                       arrowhead=1, ax=0, ay=-40)
    fig.update_layout(title_font_size=22, xaxis_title='Month-Year',
                      yaxis_title='Number of Mistakes', font=dict(color='white'),
                      plot_bgcolor='black', paper_bgcolor='black',
                      hovermode='x unified')
    return fig

def plot_event_summary_interactive():
    df1 = df.copy()
    df1['StartDate'] = pd.to_datetime(df1['StartDate'], errors='coerce')
    df1 = df1.dropna(subset=['StartDate', 'Event'])

    # Time groupings
    df1['Game'] = df1['Event']
    df1['Weekly'] = df1['StartDate'].dt.to_period('W').astype(str)
    df1['Monthly'] = df1['StartDate'].dt.to_period('M').astype(str)
    df1['Yearly'] = df1['StartDate'].dt.year.astype(str)

    grouped_data = {}
    for period in ['Game', 'Weekly', 'Monthly', 'Yearly']:
        summary = df1.groupby(period).agg(
            ActualYellow=('Yellow ', 'sum'),
            ActualCorner=('Corner', 'sum'),
            StatisticianMistakes=('Statistician (Adjusted) Name', 'nunique')
        ).reset_index().sort_values(period)
        grouped_data[period] = summary

    # Start with Weekly
    default_mode = 'Weekly'
    fig = go.Figure()

    for i, (mode, data) in enumerate(grouped_data.items()):
        visible = (mode == default_mode)

        fig.add_trace(go.Bar(
            x=data[mode],
            y=data['ActualCorner'],
            name='Actual Corners',
            marker_color='red',
            opacity=0.8,
            visible=visible
        ))
        fig.add_trace(go.Bar(
            x=data[mode],
            y=data['ActualYellow'],
            name='Actual Yellow Cards',
            marker_color='yellow',
            opacity=0.8,
            visible=visible
        ))
        fig.add_trace(go.Scatter(
            x=data[mode],
            y=data['StatisticianMistakes'],
            mode='lines',
            name='Statistician Mistakes',
            line=dict(color='lime', width=1.2, dash='dot'),
            visible=visible
        ))

    # Buttons
    buttons = []
    labels = ['All Games', 'Weekly', 'Monthly', 'Yearly']
    for i, label in enumerate(labels):
        visibility = [False] * 12
        visibility[i * 3:i * 3 + 3] = [True, True, True]
        buttons.append(dict(
            label=label,
            method='update',
            args=[
                {"visible": visibility},
                {"title": f"{label} Event Breakdown with Statistician Overlay"}
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            buttons=buttons,
            direction='right',
            showactive=False,
            x=0.00,
            y=1.1,
            xanchor='left',
            yanchor='top',
            bgcolor='black',         
            bordercolor='black',
            font=dict(color='white')  
        )],
        barmode='stack',
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
        title='Weekly Event Breakdown with Statistician Overlay',
        xaxis=dict(
            title='Time',
            showticklabels=False,
            tickmode='linear',
            ticks='',
            showgrid=False
        ),
        yaxis_title='Event Count'
    )

    return fig

def plot_mistakes_vs_match_intensity():
    df_ = df.copy()
    df_ = df_.rename(columns={'Yellow ': 'Yellow'})
    df_['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
    df_ = df_.dropna(subset=['StartDate'])

    # Match-level grouping
    game_summary = df_.groupby(['StartDate', 'Event', 'Country']).agg(
        Total_Yellow=('Yellow', 'sum'),
        Total_Corner=('Corner', 'sum'),
        Mistake_Count=('MistakeType', 'count'),
        Unique_Statisticians=('Statistician (Adjusted) Name', pd.Series.nunique)
    ).reset_index()

    # Add match intensity
    game_summary['Match_Intensity'] = game_summary['Total_Yellow'] + game_summary['Total_Corner']

    # Create figure
    fig = go.Figure()

    for country, color in zip(['England', 'Scotland'], ['orange', 'skyblue']):
        subset = game_summary[game_summary['Country'] == country]

        fig.add_trace(go.Scatter(
            x=subset['Match_Intensity'],
            y=subset['Mistake_Count'],
            mode='markers',
            name=f"{country}",
            marker=dict(size=subset['Unique_Statisticians'] * 3, color=color, opacity=0.7),
            hovertemplate='Intensity: %{x}<br>Mistakes: %{y}<br>Statisticians: %{marker.size}<extra></extra>'
        ))

        if len(subset) >= 2:
            coeffs = np.polyfit(subset['Match_Intensity'], subset['Mistake_Count'], deg=1)
            x_range = np.linspace(subset['Match_Intensity'].min(), subset['Match_Intensity'].max(), 100)
            y_pred = np.polyval(coeffs, x_range)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name=f"{country} Trend",
                line=dict(dash='dot', width=2, color=color)
            ))

    # Layout
    fig.update_layout(
        title='Mistakes vs Match Intensity with Statistician Complexity Overlay',
        xaxis_title='Match Intensity (Yellow + Corners)',
        yaxis_title='Number of Mistakes',
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    return fig

def plot_mistake_rate_per_100_events():

    df_2 = df.copy()
    df_2 = df_2.rename(columns={'Yellow ': 'Yellow'})
    df_2['StartDate'] = pd.to_datetime(df_2['StartDate'], errors='coerce')
    df_2 = df_2.dropna(subset=['StartDate'])

    # Aggregate game-level data
    game_summary = df_2.groupby(['StartDate', 'Event', 'Country']).agg(
        Total_Yellow=('Yellow', 'sum'),
        Total_Corner=('Corner', 'sum'),
        Mistake_Count=('MistakeType', 'count')
    ).reset_index()

    # Compute total events and normalised mistake rate
    game_summary['Total_Events'] = game_summary['Total_Yellow'] + game_summary['Total_Corner']
    game_summary = game_summary[game_summary['Total_Events'] > 0]
    game_summary['Mistakes_per_100_Events'] = (game_summary['Mistake_Count'] / game_summary['Total_Events']) * 100

    # Plot box + scatter
    fig = px.box(
        game_summary,
        x='Country',
        y='Mistakes_per_100_Events',
        color='Country',
        points='all',
        title='Mistake Rate per 100 Events by Country',
        template='plotly_dark',
        labels={'Mistakes_per_100_Events': 'Mistakes per 100 Events'}
    )

    fig.update_layout(
        title_font_size=20,
        font=dict(color='white'),
        plot_bgcolor='black',
        paper_bgcolor='black'
    )

    return fig


def plot_mistake_forecast():
    df_prophet = df.copy()
    df_prophet['StartDate'] = pd.to_datetime(df_prophet['StartDate'])
    ts_data = df_prophet[df_prophet['MistakeSeverity'] == 'Moderate'].copy()

    # Monthly aggregation
    ts_data['Month'] = ts_data['StartDate'].dt.to_period('M').dt.to_timestamp()
    monthly_counts = ts_data.groupby('Month').size().reset_index(name='Mistakes')

    # Reindex for continuity
    full_months = pd.date_range(
        start=monthly_counts['Month'].min(),
        end=monthly_counts['Month'].max(),
        freq='MS'
    )
    monthly_counts = monthly_counts.set_index('Month').reindex(full_months).fillna(0).reset_index()
    monthly_counts.columns = ['ds', 'y']

    # Forecast
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.add_country_holidays(country_name='UK')
    model.fit(monthly_counts)

    future = model.make_future_dataframe(periods=24, freq='M')
    forecast = model.predict(future)

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_counts['ds'],
        y=monthly_counts['y'],
        mode='lines',
        name='Actual Mistakes',
        line=dict(color='yellow', width=3, dash='dot'),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 255, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['trend'],
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Monthly Football Match Mistakes Forecast with Trend Analysis',
        xaxis_title='Month',
        yaxis_title='Number of Mistakes',
        template='plotly_white',
        hovermode='x unified',
        height=600,
        xaxis=dict(
            tickformat='%b %Y',
            dtick='M12',
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        legend=dict(
            x=0.00,
            y=0.99,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=12)
        )
    )

    return fig

def plot_shap_analysis():
    df_shap = df.copy()

    # Select relevant features and target
    features = ['Competition', 'Country', 'Event', 'Yellow ', 'Corner']
    target = 'MistakeType'

    data = df_shap[features + [target]].dropna()
    data = data[data[target].isin(['Corner', 'Yellow Card'])].copy()

    # Label encode
    le = LabelEncoder()
    data['target_encoded'] = le.fit_transform(data[target])

    # Preprocessing and model pipeline
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Competition', 'Country', 'Event'])
    ], remainder='passthrough')

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42))
    ])

    X = data[features]
    y = data['target_encoded']
    model.fit(X, y)

    # SHAP
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    preprocessed_data = model.named_steps['preprocessor'].transform(X)
    shap_values = explainer.shap_values(preprocessed_data)

    # Prepare plot data
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    shap_df = pd.DataFrame({
        'Feature': np.repeat(feature_names, X.shape[0]),
        'SHAP Value': shap_values.reshape(-1),
        'Feature Value': preprocessed_data.toarray().reshape(-1)
    })
    shap_df['Feature'] = shap_df['Feature'].str.replace(r'^cat_', '', regex=True)
    shap_df['Feature'] = shap_df['Feature'].str.replace('_', ' ')
    shap_df['Feature'] = shap_df['Feature'].apply(lambda x: f"<i>{x}</i>")

    summary_df = shap_df.groupby('Feature').agg(
        Mean_SHAP=('SHAP Value', 'mean'),
        Avg_Feature_Value=('Feature Value', 'mean')
    ).reset_index().sort_values('Mean_SHAP', ascending=True)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=summary_df['Feature'],
        x=summary_df['Mean_SHAP'],
        orientation='h',
        marker=dict(
            color=summary_df['Mean_SHAP'],
            colorscale='Tealgrn',
            colorbar=dict(title='SHAP Value Impact')
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean SHAP Value: %{x:.2f}<br>"
            "Avg Feature Value: %{customdata:.2f}<extra></extra>"
        ),
        customdata=summary_df['Avg_Feature_Value']
    ))

    fig.add_vline(x=0, line=dict(width=1, dash="dash", color="black"))

    fig.update_layout(
        title=dict(
            text="<b>Football Match Mistake Analysis</b><br><sup>Feature Impact on Yellow Cards vs Corners</sup>",
            x=0.03, y=0.95,
            font=dict(size=24, family="Arial Black")
        ),
        xaxis_title="SHAP Value Impact (Average Log-Odds)",
        yaxis_title="Feature",
        template="plotly_white",
        height=800,
        margin=dict(l=150, r=50, t=150, b=50),
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Rockwell"),
        annotations=[
            dict(
                text="Source: Your Dataset | Visualization by SHAP",
                x=1, y=-0.15, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="grey")
            )
        ]
    )

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
    fig.update_yaxes(tickfont=dict(size=12))

    fig.add_annotation(x=0.15, y=1.07, xref="paper", yref="paper",
                       text="➤ Increases Yellow Card likelihood",
                       showarrow=False, font=dict(color="green", size=12))

    fig.add_annotation(x=0.15, y=1.10, xref="paper", yref="paper",
                       text="➤ Increases Corner likelihood ",
                       showarrow=False, font=dict(color="coral", size=12))

    return fig


# ------------------------ STREAMLIT APP --------------------------

#st.set_page_config(layout="wide", page_title="Football Mistake Dashboard")


# Sidebar navigation for EDA
plot_names = {
    "📈 Yearly Trend": plot_annual_mistake_trend,
    "🟥 Mistake Type Yearly Dist": plot_yearly_mistake_type_distribution,
    "🍩 Mistake Types": plot_mistake_type_distribution,
    "⏱️ Delays": plot_delay_duration_frequency,
    "⚽ Yellow vs Mistakes": plot_bubble_chart,
    "🏆 Top Competitions": plot_top_competitions,
    "🧑‍💻 Top Statisticians": plot_top_statisticians,
    "📅 Monthly Mistakes": plot_monthly_mistake_trend,
    "🌍 England vs Scotland": plot_geographic_trend,
    "📊 Actual Events + Stat Overlay": plot_event_summary_interactive,
    "🧮 Mistake Rate - Normalised": plot_mistake_rate_per_100_events,
    "📉 Mistakes vs Match Intensity": plot_mistakes_vs_match_intensity
}

# Sidebar navigation for ML
plot_names_ml = {
    "📈 Forecast (Prophet)": plot_mistake_forecast,
    "🧠 SHAP Feature Impact": plot_shap_analysis
}

# Choose section: EDA or ML

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("📌 Dashboard Sections")
    section = st.selectbox("Choose Section", ["EDA", "ML Models"])
    st.markdown("<br>", unsafe_allow_html=True)
    if section == "EDA":
        st.subheader("📊 EDA Charts")
        selected_plot = st.radio("Choose Visualisation", list(plot_names.keys()))
    else:
        st.subheader("🤖 ML Models")
        selected_plot = st.radio("Choose Model", list(plot_names_ml.keys()))

# Add logo fixed at bottom inside the sidebar
with st.sidebar:
    st.markdown(
        f"""
        <style>
            .fixed-logo {{
                position: fixed;
                bottom: 20px;
                left: 8px;
                width: 240px;
                text-align: center;
            }}
            .fixed-logo img {{
                width: 90px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                border-radius: 8px;
            }}
            .fixed-logo p {{
                font-size: 12px;
                color: grey;
                margin-top: 6px;
            }}
        </style>

        <div class="fixed-logo">
            <img src="data:image/png;base64,{logo}">
            <p>University of Strathclyde</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Show KPI bar
show_kpis(df)

# Add spacing between KPI and the chart
st.markdown("<br><br>", unsafe_allow_html=True)

# Show only the selected chart from the selected section
if section == "EDA":
    fig = plot_names[selected_plot]()
    st.plotly_chart(fig, use_container_width=True)

    # Add chart description under EDA charts
    with st.expander("📘 Chart Description"):
        if selected_plot == "📈 Yearly Trend":
            st.write("The chart titled “Annual Mistake Trends (Overall vs England vs Scotland)” illustrates the year-wise comparison of recorded mistakes across England, Scotland, and combined totals from 2022 to 2024. The overall trend, shown with a thick green line, reflects a steady increase in total mistakes each year, suggesting a general rise in data capture or reporting challenges. England’s trend, represented by a red dotted line, shows a slight dip in 2023 before climbing again in 2024, while Scotland’s line, in blue and also dotted, reveals a smaller but consistent upward trend. Despite these increases, Scotland’s mistake counts remain significantly lower than England’s, reinforcing geographical differences in reporting volume or match activity. The use of dotted lines for individual countries makes it easier to contrast their trends with the solid overall trajectory. For analysts, this visual provides valuable insight into where support, training, or process enhancements might be prioritised. It also allows stakeholders to explore whether broader increases are due to growing match volume, heightened scrutiny, or evolving data definitions across regions.")
        elif selected_plot == "🟥 Mistake Type Yearly Dist":
            st.write("The chart titled “Yearly Distribution of MistakeTypeCategory” displays the frequency of various mistake categories recorded from 2022 to 2024. Each group of bars represents a specific year, with mistake types differentiated by a gradient of red shades—from light pink for minor issues like confirmation delays to deep red for more severe timing-related risks. This consistent colour palette allows easy comparison without overwhelming the visual space. Notably, some mistake types like “Delayed” and “Missed (Updated by the LFA without the ST)” appear consistently across all years, suggesting persistent operational challenges. In contrast, categories like “Wrong use of Corner Risk” and “Delayed Corner with more than 30s in Corner Risk” show noticeable spikes or declines, indicating emerging or resolved issues. This yearly comparison helps statisticians and quality leads identify recurring pain points and prioritise categories that require targeted training, process changes, or tool enhancements. By visualising shifts in error types over time, the chart serves as both a performance tracker and a decision-making tool to improve data accuracy in future seasons.")
        elif selected_plot == "🍩 Mistake Types":
            st.write("The chart titled “Distribution of Mistake Types” illustrates the proportion of two key error categories: Yellow Card and Corner. It shows that 57.6% of all recorded moderate mistakes were related to Yellow Card events, while 42.4% were associated with Corners. This distribution highlights the relative difficulty statisticians face when recording yellow card incidents during live matches. Yellow cards are often issued unpredictably and can be influenced by rapid player actions, referee positioning, and off-ball incidents, making them harder to capture in real time. In contrast, corners tend to be more structured and easier to anticipate, yet still account for a substantial portion of errors. For statisticians, this insight is valuable for identifying which event types may require additional focus during training or system enhancements. It also supports further investigation into whether certain competitions or match conditions contribute more to specific mistake types. Overall, this chart helps to prioritise areas where interventions or support could reduce error rates and improve live data accuracy.")
        elif selected_plot == "⏱️ Delays":
            st.write("The chart titled “Delay Duration Frequency Analysis” presents the number of occurrences across different delay time categories during football match data entry. The most common delay is between 4 to 9 seconds, accounting for the highest number of instances. This is followed by delays between 10 to 30 seconds and those exceeding 60 seconds, indicating that a significant number of mistakes fall beyond the acceptable response window. Shorter delays are more frequent, suggesting that many errors may result from slight timing lapses rather than complete oversight. The presence of longer delays, however, raises concerns about potential distractions, system lags, or complex match situations that demand more time for decision-making. For statisticians, this chart highlights the need to focus on minimising short delays through practice and precision while also examining conditions that may cause extended input gaps. It also supports the argument that not all delays reflect negligence, but may instead reflect real-world challenges in live match environments. Understanding this distribution helps in setting more realistic expectations, refining thresholds for acceptable delay, and exploring supportive tools or staffing models to reduce pressure during high-event periods.")
        elif selected_plot == "⚽ Yellow vs Mistakes":
            st.write("The chart titled “Mistakes vs Yellow Cards (Bubble Size = Corners)” illustrates the relationship between the number of yellow cards in a match and the number of mistakes recorded by statisticians, with bubble size representing the number of corners. Each bubble indicates a specific event count, where larger and more vibrant bubbles correspond to matches with a higher number of corners. The chart shows that as the number of yellow cards increases, there is a tendency for mistakes to rise slightly, though not in a strictly linear fashion. Notably, some of the highest mistake counts appear in matches with both high yellow card numbers and larger corner volumes. This suggests that matches with a high level of on-field activity—measured through cards and corners—may contribute to greater cognitive demand on statisticians. The spread of data points also highlights variability in how statisticians perform under different match conditions, with some low-yellow-card matches still producing multiple mistakes. Overall, this visual underscores the potential link between match intensity and error likelihood, encouraging further analysis of how event volume and complexity affect live data accuracy. For statisticians, it supports the case for tailored strategies in high-pressure games, such as extra training or shared responsibilities.")
        elif selected_plot == "🏆 Top Competitions":
            st.write("The chart titled “Top Competitions by Number of Mistakes” highlights which football competitions experienced the highest number of recorded mistakes by statisticians. The 2023/2024 England League 1 stands out with the most mistakes, significantly higher than other competitions. This is followed by the 2022/2023 England League Two and the 2022/2023 England League 1, suggesting a consistent trend of higher error rates within lower-tier English leagues. Several Scottish leagues and national competitions also appear but with relatively fewer mistakes. The distribution suggests that certain leagues may present more challenges, possibly due to higher match intensity, reduced visibility at smaller venues, less predictable gameplay, or even fewer experienced statisticians assigned. For statisticians and analysts, this insight can help target specific competitions for review, additional training, or resource support. It also provides useful evidence when questioning whether all leagues should be held to the same performance standard. Understanding the environments where mistakes occur more frequently can inform smarter planning and more tailored support to reduce future error rates.")
        elif selected_plot == "🧑‍💻 Top Statisticians":
            st.write("The chart titled “Top Statisticians by Number of Mistakes” ranks individual statisticians based on the number of moderate mistakes they have recorded. It shows a group of statisticians at the top with 11 mistakes each, followed by others with 10, 9, and 8 mistakes respectively. The colour gradient ranges from blue to red, providing a quick visual indicator of relative performance. While the chart may initially suggest underperformance, it’s important to interpret these numbers with care. A higher mistake count may reflect greater exposure — for example, statisticians who cover more matches or more complex fixtures are more likely to make errors simply due to volume and intensity. This chart is most useful when combined with other metrics such as number of games covered, events per match, or days of inactivity before the game. For statisticians, the visual can act as a prompt for reflective analysis rather than blame. It encourages deeper review of working patterns, support systems, and whether workloads are distributed fairly across individuals. Used appropriately, this insight supports constructive performance discussions and helps inform training or workload balancing efforts aimed at reducing mistake frequency across the team.")
        elif selected_plot == "📅 Monthly Mistakes":
            st.write("The chart titled “Monthly Trend of Mistakes Across All Competitions” illustrates the number of recorded mistakes per month from January 2022 to late 2024. It reveals a strong seasonal pattern, with noticeable peaks and troughs aligning with the football calendar. Mistakes tend to rise during periods of high match activity, such as winter and early spring, and drop sharply during summer months, which typically coincide with off-season or reduced fixtures. Notably, the overall level of mistakes appears to rise slightly year-on-year, with the highest spikes occurring in early 2024. This chart provides useful insight for statisticians and operations teams by identifying when workloads and error risks are highest. It supports the idea that mistake frequency may be influenced more by match volume and intensity than individual performance alone. These trends can help guide staffing plans, training schedules, and support mechanisms around peak periods, ensuring statisticians are better prepared when demand is highest. Additionally, it strengthens the case for reviewing static KPI thresholds in light of seasonal fluctuations, helping build a fairer performance evaluation model.")
        elif selected_plot == "🌍 England vs Scotland":
            st.write("The chart titled “Geographic Impact: Mistakes Over Time (England vs Scotland)” compares the monthly number of recorded mistakes in England and Scotland from early 2022 to late 2024. A clear distinction is observed: England consistently reports a higher number of mistakes than Scotland throughout the period. A red vertical dashed line highlights the introduction of VAR in Scotland, which occurred in October 2023. Interestingly, the implementation of VAR appears to coincide with a slight increase in Scotland’s mistake counts in the following months, though they remain significantly lower than England’s. The consistent gap between the two countries suggests that environmental or structural differences may influence data collection challenges. Factors could include differences in league intensity, match pace, staffing models, or even match complexity. The VAR marker also adds an important dimension, helping analysts consider how technological changes might alter the statistician’s workload or timing. For statisticians and decision-makers, this visual reinforces the need to interpret mistake trends in the context of local league structures and operational realities. It also opens up questions about how tools like VAR may impact accuracy and whether support models should differ between countries or match levels.")
        elif selected_plot == "📊 Actual Events + Stat Overlay":
            st.write("The chart titled “All Games Event Breakdown with Statistician Overlay” presents a detailed view of actual game events across the full dataset, with a focus on yellow cards, corners, and statistician mistakes. Each thin vertical bar represents a single football game, with red indicating actual corners and yellow stacked above for actual yellow cards, allowing a quick assessment of total events per match. Overlaid on the bars is a very lean dotted green line showing the number of statistician mistakes per game, helping to track error trends alongside match intensity. Users can switch between weekly, monthly, yearly, and all-game groupings using the buttons at the top-left corner of the chart, offering flexible exploration of data density over time. The visual highlights the variability in game events and suggests a potential relationship between high-event matches and error counts. Particularly dense matches often align with slight rises in statistician mistakes, which may imply cognitive load challenges during busy fixtures. By combining detailed event counts with overlayed error trends, this chart enables analysts and coordinators to spot patterns in workload pressure, helping to guide support planning and error reduction strategies for future fixtures.")
        elif selected_plot == "🧮 Mistake Rate - Normalised":
            st.write("The chart titled “Mistake Rate per 100 Events by Country“ presents a normalised view of errors by comparing the number of mistakes made per 100 in-game events (yellow cards and corners) in England and Scotland. This box-and-scatter plot reveals the consistency and extremity of performance between countries. Scotland shows a tighter distribution with fewer outliers and a lower overall spread, suggesting greater consistency and control in mistake handling per match intensity. In contrast, England not only has a broader range but also a number of high outliers, including cases where mistake rates spike dramatically—indicating potential issues during high-pressure games. The central tendency (median) between the two is relatively similar, but the presence of extreme outliers in England skews the perception of mistake risk. This normalised comparison supports a data-backed conclusion: while both countries may operate under similar base conditions, England’s mistake profile includes more volatile scenarios, likely triggered by either extreme match conditions or inconsistent operational execution. This makes the case for a deeper review of high-mistake matches to identify root causes.")
        elif selected_plot == "📉 Mistakes vs Match Intensity":
            st.write("The chart titled “Mistakes vs Match Intensity with Statistician Complexity Overlay“ visualises how the number of mistakes correlates with match intensity—defined as the total number of yellow cards and corners—across England and Scotland. Each point represents a game, with the size of the point indicating how many statisticians were involved. The dotted regression lines show the trend for each country. Interestingly, Scotland exhibits a steeper trend line, suggesting that mistakes increase more rapidly with intensity compared to England. This may imply that Scotland’s mistake count is more sensitive to rising match pressure, potentially due to fewer resources, less experience, or operational factors. However, the number of statisticians involved (point size) does not dramatically differ, hinting that errors may not be purely due to workload but could relate to how each region manages fast-paced or complex matches. This chart helps decision-makers assess whether mistake patterns stem more from match dynamics or the supporting personnel model.")

else:
    fig = plot_names_ml[selected_plot]()
    st.plotly_chart(fig, use_container_width=True)

    # Add chart description under ML model outputs
    with st.expander("📘 Chart Description"):
        if selected_plot == "📈 Forecast (Prophet)":
            st.write("The chart titled “Monthly Football Match Mistakes Forecast with Trend Analysis” combines historical mistake data with a forward-looking forecast generated using the Prophet model. The blue line represents actual monthly mistakes, while the green line shows predicted future values with shaded bands indicating the confidence interval. A dotted red line highlights the long-term trend. The forecast projects that mistakes will continue to fluctuate with seasonal peaks, maintaining an overall slightly upward trajectory over time. The confidence interval widens further into the future, reflecting greater uncertainty in longer-term predictions. Notably, the historical data shows recurring spikes, often aligned with peak football periods such as winter and spring. This seasonal variation is effectively captured in the forecast, supporting the reliability of the model. For statisticians and operational leads, this chart is a useful planning tool, offering evidence-based insight into when mistake volumes are likely to rise. It can help with resource allocation, training cycles, and reviewing the realism of performance KPIs. By combining past behaviour with future expectations, the visual strengthens the case for proactive rather than reactive decision-making in managing live data quality across the football calendar.")
        elif selected_plot == "🧠 SHAP Feature Impact":
            st.write("The chart titled “Football Match Mistake Analysis – Feature Impact on Yellow Cards vs Corners” presents a SHAP (SHapley Additive exPlanations) summary plot that measures how individual features contribute to the classification of mistakes as either related to yellow cards or corners. Each feature, such as specific events, competitions, or matches, is plotted against its SHAP value, which represents its average impact on the model's decision. Features to the right of zero (positive SHAP values) increase the likelihood of a mistake being classified as a yellow card, while those to the left (negative SHAP values) lean towards corner-related mistakes.This interpretability tool is crucial for statisticians and analysts, as it goes beyond correlation and offers a transparent view of which match contexts or events are most associated with specific mistake types. For instance, certain high-intensity fixtures or competitions like England League Cup or EFL Trophy show notable influence. The model suggests that event-driven complexity can shift the likelihood of mistakes depending on the match profile. By understanding this breakdown, teams can identify high-risk fixtures for targeted intervention, refine training efforts for frequently problematic contexts, and ultimately reduce classification errors. This chart supports data-driven reasoning and accountability in live data environments.")

