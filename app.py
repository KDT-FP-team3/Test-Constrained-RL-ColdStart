import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from agent import SP500Environment, RecommendationAgent

# == [UI 개선] 메트릭 라벨 색상 강제 지정 CSS ==
st.markdown("""
<style>
/* 첫 번째 컬럼(Vanilla) 라벨 붉은색 */
div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetricLabel"] { color: red !important; font-weight: bold !important; font-size: 1.1rem !important; }
/* 두 번째 컬럼(STATIC) 라벨 파란색 */
div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetricLabel"] { color: blue !important; font-weight: bold !important; font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# == 대시보드 초기 설정 ==
st.set_page_config(page_title="Test-Constrained-RL", layout="wide")

st.markdown("## >> Test-Constrained-RL-ColdStart: S&P 500 Performance")

# == Trial History (시도 누적 기록) 초기화 ==
if 'trial_history' not in st.session_state:
    st.session_state.trial_history = []

# == 데이터 환경 및 에이전트 초기화 ==
with st.spinner('>> 실시간 S&P 500 데이터를 분석 중입니다... (약 10초 소요)'):
    env = SP500Environment()
    agent_raw = RecommendationAgent(env, use_constraints=False)
    agent_static = RecommendationAgent(env, use_constraints=True)

# == 차트 및 UI 제어 사이드바 ==
st.sidebar.markdown("### >> Test Parameters")
max_episodes = len(env.data) - 20 - 1 if len(env.data) > 20 else 100
episodes = st.sidebar.slider("Episodes (Trading Days)", 10, max_episodes, min(100, max_episodes))
speed = st.sidebar.slider("Frame Speed (sec)", 0.0, 0.5, 0.05)

# == Plotly 차트: 마커 및 기준선, 투명 배경 적용 ==
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[0], y=[0], mode='lines+markers', name='<b>Vanilla RL (Unconstrained)</b>', 
    line=dict(color='red', width=3), marker=dict(symbol='circle-open', size=8, line_width=2)
))

fig.add_trace(go.Scatter(
    x=[0], y=[0], mode='lines+markers', name='<b>RL with STATIC (Ours)</b>', 
    line=dict(color='blue', width=3), marker=dict(symbol='square-open', size=8, line_width=2)
))

fig.update_layout(
    title=dict(text="<b>Cumulative Return Comparison (S&P 500)</b>", font=dict(size=28, color='black')),
    xaxis=dict(title="<b>Trading Days</b>", titlefont=dict(size=22, color='black'), showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title="<b>Total Cumulative Return (%)</b>", titlefont=dict(size=22, color='black'), showgrid=True, gridcolor='lightgray'),
    # !! [개선] 범례 배경을 투명(rgba(0,0,0,0))하게 설정
    legend=dict(font=dict(size=20, color='black'), x=0.01, y=0.99, borderwidth=1, bgcolor='rgba(0,0,0,0)'),
    plot_bgcolor='white', height=500
)

fig.add_hline(y=0, line_width=3, line_color="black", opacity=0.8)

chart_view = st.empty()
chart_view.plotly_chart(fig, use_container_width=True)

# == 실시간 지표 카드 영역 ==
col1, col2 = st.columns(2)
m_u = col1.empty()
m_s = col2.empty()

st.markdown("---")
analysis_view = st.empty()

if st.button(">> Run Evaluation"):
    h_u, h_s, steps = [0], [0], [0]
    log_data = []

    for i in range(20, 20 + episodes):
        ticker_u, _, r_u = agent_raw.select_action(current_step=i)
        ticker_s, valid_s, r_s = agent_static.select_action(current_step=i)
        
        h_u.append(h_u[-1] + r_u)
        h_s.append(h_s[-1] + r_s)
        current_day = i - 19
        steps.append(current_day) 
        
        log_data.append({"Day": current_day, "Vanilla Pick": ticker_u, "Vanilla Return(%)": round(r_u, 2), "STATIC Pick (Ours)": ticker_s, "STATIC Return(%)": round(r_s, 2)})
        
        fig.data[0].x = steps; fig.data[0].y = h_u
        fig.data[1].x = steps; fig.data[1].y = h_s
        chart_view.plotly_chart(fig, use_container_width=True)
        
        # !! [개선] <b> 태그 제거. 색상은 상단의 CSS가 자동으로 입혀줌
        m_u.metric(label="Unconstrained Return", value=f"{h_u[-1]:.2f}%", delta=f"{r_u:.2f}%")
        m_s.metric(label=f"STATIC Return - Bought: {ticker_s}", value=f"{h_s[-1]:.2f}%", delta=f"{r_s:.2f}%")
        
        time.sleep(speed)

    # == 시뮬레이션 종료 후 Session State에 결과 저장 ==
    trial_num = len(st.session_state.trial_history) + 1
    st.session_state.trial_history.append({
        "Trial": trial_num,
        "Vanilla Final (%)": round(h_u[-1], 2),
        "STATIC Final (%)": round(h_s[-1], 2)
    })

    df_log = pd.DataFrame(log_data)
    with analysis_view.container():
        st.markdown("#### >> Agent Decision Analysis")
        col3, col4 = st.columns([1.2, 1])
        with col3:
            st.dataframe(df_log.set_index("Day"), height=300, use_container_width=True)
        with col4:
            dist_counts = df_log['STATIC Pick (Ours)'].value_counts().reset_index()
            dist_counts.columns = ['Ticker', 'Buy Count']
            fig_bar = px.bar(dist_counts, x='Ticker', y='Buy Count', title="Frequency of Safe-Asset Selection", color='Buy Count', color_continuous_scale='Blues')
            fig_bar.update_layout(plot_bgcolor='white', height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

# == 하단: 누적 Trial History 차트 및 테이블 ==
if len(st.session_state.trial_history) > 0:
    st.markdown("---")
    st.markdown("### 🏆 Trial History: STATIC vs Vanilla")
    st.markdown("Run Evaluation을 반복할수록 두 모델의 최종 누적 수익률 격차 추이를 확인할 수 있습니다.")
    
    history_df = pd.DataFrame(st.session_state.trial_history)
    
    col_hist_chart, col_hist_table = st.columns([2, 1])
    
    with col_hist_chart:
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(x=history_df['Trial'], y=history_df['Vanilla Final (%)'], mode='lines+markers', name='Vanilla', line=dict(color='red')))
        fig_history.add_trace(go.Scatter(x=history_df['Trial'], y=history_df['STATIC Final (%)'], mode='lines+markers', name='STATIC', line=dict(color='blue')))
        fig_history.update_layout(xaxis_title="Trial Number", yaxis_title="Final Cumulative Return (%)", plot_bgcolor='white', height=350, legend=dict(bgcolor='rgba(0,0,0,0)'))
        fig_history.add_hline(y=0, line_width=2, line_color="black", opacity=0.8)
        st.plotly_chart(fig_history, use_container_width=True)
        
    with col_hist_table:
        st.dataframe(history_df.set_index("Trial"), height=350, use_container_width=True)