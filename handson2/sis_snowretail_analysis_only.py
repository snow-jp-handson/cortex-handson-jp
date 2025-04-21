# =========================================================
# ↑上記で、Plotly, snowflake,core, snowflake-ml-pythonの追加が必要
# =========================================================

# =========================================================
# 必要なライブラリのインポート
# =========================================================
# 基本ライブラリ
import streamlit as st
import pandas as pd
import json

# Streamlitの設定
st.set_page_config(layout="wide")

# 可視化ライブラリ
import plotly.express as px

# Snowflake関連ライブラリ
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete as CompleteText
from snowflake.core import Root

# =========================================================
# 定数定義
# =========================================================
# Cortex Analyst APIの設定
ANALYST_API_ENDPOINT = "/api/v2/cortex/analyst/message"
ANALYST_API_TIMEOUT = 50  # 秒

# =========================================================
# Snowflake接続と共通ユーティリティ関数
# =========================================================

# Snowflakeセッションの取得
snowflake_session = get_active_session()

def get_available_warehouses() -> list:
    """利用可能なSnowflakeウェアハウスの一覧を取得します。
    
    Returns:
        list: ウェアハウス名のリスト（取得失敗時は空リスト）
    """
    try:
        result = snowflake_session.sql("SHOW WAREHOUSES").collect()
        return [row['name'] for row in result]
    except Exception as e:
        st.error(f"ウェアハウスの取得に失敗しました: {str(e)}")
        return []

# =========================================================
# メイン処理
# =========================================================

# メインコンテンツ
st.title("🏪 スノーリテール 社内アプリ")
st.markdown("---")

"""Cortex Analystを使用した分析チャットボットページを表示します。"""
st.header("分析チャットボット")

# ワークショップ向けの説明
st.info("""
## 📊 分析チャットボットについて

このページでは、Snowflake Cortex Analystを使用したデータ分析チャットボットを体験できます。

### 主な機能
* **自然言語でのデータ分析**: 質問をSQLに自動変換し、データベースに対して実行
* **視覚化**: 分析結果を自動的にグラフ化して表示
* **日本語対応**: 英語で返される分析結果を自動的に日本語に翻訳

### 使い方のヒント
* 今回は店舗とECの取引データを元にセマンティックモデルを作成しているため、販売数量や売上金額などの分析に適しています。
* データに関する質問を具体的に記述してください（例：「2023年の四半期ごとの売上推移を教えて」）
* 質問はデータに関連するものに限定されます（一般的な会話ではなく、データ分析のクエリとして解釈されます）
* 分析結果はグラフと表形式で表示され、生成されたSQLも確認できます

### セマンティックモデル
このチャットボットは、事前に作成されたセマンティックモデルを使用してデータベースのスキーマを理解しています。
""")

# セマンティックモデルの設定情報
SEMANTIC_MODEL_STAGE = "SEMANTIC_MODEL_STAGE"
SEMANTIC_MODEL_FILE = "sales_analysis_model.yaml"
full_stage_path = f"@{SEMANTIC_MODEL_STAGE}/{SEMANTIC_MODEL_FILE}"

# セッション状態の初期化
if "analyst_messages" not in st.session_state:
    st.session_state.analyst_messages = []

# チャット履歴のクリアボタン
if st.button("チャット履歴をクリア"):
    st.session_state.analyst_messages = []
    st.rerun()

# チャット履歴の表示
for message in st.session_state.analyst_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message and message["result"] is not None:
            # 結果が DataFrame であれば表示
            if isinstance(message["result"], pd.DataFrame) and not message["result"].empty:
                st.dataframe(message["result"])
            
            # SQLクエリが含まれていれば表示
            if "sql" in message and message["sql"]:
                with st.expander("生成されたSQL"):
                    st.code(message["sql"], language="sql")
            
            # グラフが含まれていれば表示
            if "chart" in message and message["chart"]:
                st.plotly_chart(message["chart"], use_container_width=True, key=f"analyst_chart_{st.session_state.analyst_messages.index(message)}")

# ユーザー入力の処理
if prompt := st.chat_input("データについて質問してください"):
    # ユーザーメッセージの表示
    st.session_state.analyst_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 回答生成の処理
    with st.spinner("回答を生成中..."):
        try:
            # セマンティックモデルファイルの存在確認
            file_exists = False
            try:
                stage_content = snowflake_session.sql(f"""
                    LIST @{SEMANTIC_MODEL_STAGE}
                """).collect()
                
                for file in stage_content:
                    if SEMANTIC_MODEL_FILE in file['name']:
                        file_exists = True
                        break
                
                if not file_exists:
                    raise Exception(f"セマンティックモデルファイル '{SEMANTIC_MODEL_FILE}' がステージ '{SEMANTIC_MODEL_STAGE}' に見つかりません。")
                
            except Exception as e:
                st.error(f"ステージの確認中にエラーが発生しました: {str(e)}")
                st.code(str(e))
                raise e
            
            # メッセージの準備
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # リクエストボディの準備
            request_body = {
                "messages": messages,
                "semantic_model_file": full_stage_path,
            }
            
            # Cortex Analyst API呼び出し
            try:
                import _snowflake
                # Snowflake内部APIを使用
                resp = _snowflake.send_snow_api_request(
                    "POST",
                    ANALYST_API_ENDPOINT,
                    {},  # headers
                    {},  # params
                    request_body,
                    None,  # request_guid
                    ANALYST_API_TIMEOUT * 1000,  # ミリ秒に変換
                )
                
                # レスポンスの処理
                if resp["status"] < 400:
                    response_data = json.loads(resp["content"])
                    if "message" in response_data and "content" in response_data["message"]:
                        content_list = response_data["message"]["content"]
                        
                        # テキストとSQLを抽出
                        response_text = ""
                        sql_query = ""
                        result_data = None
                        chart = None
                        
                        for item in content_list:
                            if item["type"] == "text":
                                response_text += item["text"] + "\n\n"
                            elif item["type"] == "sql":
                                sql_query = item["statement"]
                        
                        # 英語のレスポンスを日本語に翻訳
                        if response_text:
                            try:
                                translated_response = snowflake_session.sql("""
                                    SELECT SNOWFLAKE.CORTEX.TRANSLATE(?, 'en', 'ja') as translated
                                """, params=[response_text.strip()]).collect()[0]['TRANSLATED']
                                response_text = translated_response
                            except Exception as translate_error:
                                st.warning(f"翻訳中にエラーが発生しました。元の英語レスポンスを表示します: {str(translate_error)}")
                        
                        # SQLを実行してデータフレームを取得
                        try:
                            result_data = snowflake_session.sql(sql_query).to_pandas()
                            
                            # シンプルなグラフを作成（データに基づいて）
                            if not result_data.empty and len(result_data.columns) >= 2:
                                x_col = result_data.columns[0]
                                y_col = result_data.columns[1]
                                
                                # データタイプに基づいて適切なグラフを選択
                                if result_data[x_col].dtype == 'object':  # カテゴリデータ
                                    chart = px.bar(
                                        result_data,
                                        x=x_col,
                                        y=y_col,
                                        title="分析結果"
                                    )
                                else:  # 数値データ
                                    chart = px.line(
                                        result_data,
                                        x=x_col,
                                        y=y_col,
                                        title="分析結果"
                                    )
                        except Exception as sql_error:
                            st.error(f"SQL実行エラー: {str(sql_error)}")
                        
                        # 応答をチャット履歴に追加
                        st.session_state.analyst_messages.append({
                            "role": "assistant", 
                            "content": response_text.strip(),
                            "result": result_data,
                            "sql": sql_query,
                            "chart": chart
                        })
                        
                        # 応答を表示
                        with st.chat_message("assistant"):
                            st.markdown(response_text.strip())
                            
                            if result_data is not None and not result_data.empty:
                                st.dataframe(result_data)
                            
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                            
                            if sql_query:
                                with st.expander("生成されたSQL"):
                                    st.code(sql_query, language="sql")
                    else:
                        raise Exception("APIレスポンスの形式が不正です")
                else:
                    error_content = json.loads(resp["content"])
                    error_msg = f"""
                    🚨 APIエラーが発生しました 🚨
                    
                    * ステータスコード: `{resp['status']}`
                    * リクエストID: `{error_content.get('request_id', 'N/A')}`
                    * エラーコード: `{error_content.get('error_code', 'N/A')}`
                    
                    メッセージ:
                    ```
                    {error_content.get('message', '不明なエラー')}
                    ```
                    """
                    raise Exception(error_msg)
            
            except ImportError:
                # Snowflake内部APIが使用できない場合
                st.error("Snowflakeの内部APIにアクセスできません。Streamlit in Snowflake環境で実行してください。")
                
                # 代替のモック応答を生成（デモ用）
                response_text = f"質問: {prompt}\n\n申し訳ありませんが、Cortex Analystへのアクセスが現在利用できません。本来であれば、ここにデータ分析の結果が日本語で表示されます。"
                result_data = pd.DataFrame({
                    "カテゴリ": ["商品の品質", "価格", "接客サービス", "店舗環境", "その他"],
                    "レビュー数": [45, 32, 28, 15, 10]
                })
                
                # モックチャート
                chart = px.bar(
                    result_data,
                    x="カテゴリ",
                    y="レビュー数",
                    title="カテゴリ別レビュー数（モックデータ）"
                )
                
                # 応答をチャット履歴に追加
                st.session_state.analyst_messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "result": result_data,
                    "sql": "-- モックSQL\nSELECT category_name, COUNT(*) as count\nFROM REVIEW_TAGS\nGROUP BY category_name\nORDER BY count DESC",
                    "chart": chart
                })
                
                # 応答を表示
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                    st.dataframe(result_data)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    with st.expander("モックSQL"):
                        st.code("-- モックSQL\nSELECT category_name, COUNT(*) as count\nFROM REVIEW_TAGS\nGROUP BY category_name\nORDER BY count DESC", language="sql")
            
        except Exception as e:
            error_msg = f"""
            Cortex Analystにアクセスできません。
            エラー: {str(e)}
            
            **確認事項:**
            1. セマンティックモデルファイル '{SEMANTIC_MODEL_FILE}' がステージ '{SEMANTIC_MODEL_STAGE}' に存在するか確認してください。
            2. Cortex Analystサービスが有効になっているか確認してください。
            3. 必要な権限が付与されているか確認してください。
            
            **セマンティックモデルのパス:** {full_stage_path}
            """
            
            st.error(error_msg)
            st.code(str(e))
            
            # エラーメッセージをチャット履歴に追加
            st.session_state.analyst_messages.append({
                "role": "assistant", 
                "content": f"エラーが発生しました: {str(e)}",
                "result": None
            })