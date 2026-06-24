from __future__ import annotations
import pandas as pd, streamlit as st
st.title('Analytics')
df=st.session_state.get('clean_dataset')
if df is None: st.warning('Load a dataset from the main page.'); st.stop()
df=df.copy(); df['TransactionDate']=pd.to_datetime(df['TransactionDate'])
tabs=st.tabs(['Withdrawal Analytics','Deposit Analytics','Category Breakdown'])
with tabs[0]:
    st.line_chart(df.set_index('TransactionDate')[['Withdrawal_Amount','Rolling_7_Day_Withdrawal_Amount','Rolling_30_Day_Withdrawal_Amount']])
    st.bar_chart(df.groupby('DayOfWeek')['Withdrawal_Amount'].mean())
with tabs[1]:
    st.bar_chart(df.set_index('TransactionDate')['Deposit_Amount'])
    st.bar_chart(df.groupby('DayOfWeek')['Has_Deposit'].mean())
    st.line_chart(df.set_index('TransactionDate')['Rolling_30_Day_Deposit_Amount'])
    st.dataframe(df.groupby('Has_Deposit')[['Total_Debit','Total_Credit','Transaction_Count','Deposit_Amount']].mean())
with tabs[2]:
    cats=['Withdrawal','Deposit','Airtime','BillPayment','WalletTransfer','Transfer','Levy','Commission','Reversal','Insurance','Other']
    totals={c: float(df[f'{c}_Amount'].sum()) for c in cats}
    st.bar_chart(pd.Series(totals, name='Amount'))
