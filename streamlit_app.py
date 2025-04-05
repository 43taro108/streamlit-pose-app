# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 07:51:29 2025

@author: ktrpt
"""
import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import pandas as pd
import os
import zipfile

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="MediaPipe CSV & 動画出力", layout="centered")
st.title("MediaPipe Pose × CSV（ピクセル座標）＆骨格付き動画出力")

st.info("アップロードされた動画はサーバーに保存されません。処理後に自動で削除されます。")

# セッション初期化
for key in ["video_path", "csv_ready", "csv_data", "downloaded", "annotated_video_path"]:
    if key not in st.session_state:
        st.session_state[key] = None if "path" in key or "data" in key else False

# アップロード受付
uploaded_file = st.file_uploader("動画ファイルをアップロード（.mp4 / .mov / .avi）", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.session_state.video_path = tfile.name
    st.session_state.csv_ready = False
    st.session_state.downloaded = False
    st.session_state.annotated_video_path = None

video_path = st.session_state.video_path

# アップロード後
if video_path and os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    st.markdown(f"📊 フレーム数: **{frame_count}** | FPS: **{fps:.2f}** | サイズ: {width}×{height} | 時間: **{duration:.1f} 秒**")

    start_frame, end_frame = st.slider("✂️ 分析フレーム範囲", 0, frame_count - 1, (0, min(frame_count - 1, 100)))
    current_frame = st.slider("▶プレビュー表示フレーム", start_frame, end_frame, start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(preview, caption=f"Frame {current_frame}", use_column_width=True)
    cap.release()

    if st.button("骨格検出＋ピクセル座標で出力"):
        with st.spinner("骨格検出・動画処理中...お待ちください"):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            all_data = []

            # 出力動画の準備
            out_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            frame_idx = start_frame
            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    row = [frame_idx]
                    for lm in results.pose_landmarks.landmark:
                        x_px = lm.x * width
                        y_px = lm.y * height
                        row += [x_px, y_px]
                    all_data.append(row)

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                out.write(frame)
                frame_idx += 1

            cap.release()
            out.release()

        if all_data:
            # ヘッダー：frame, x_0_px, y_0_px, ..., x_32_px, y_32_px
            columns = ["frame"]
            for i in range(33):
                columns += [f"x_{i}_px", f"y_{i}_px"]
            df = pd.DataFrame(all_data, columns=columns)

            st.session_state.csv_data = df.to_csv(index=False).encode("utf-8")
            st.session_state.csv_ready = True
            st.session_state.annotated_video_path = out_path
            st.success(f"{len(df)} フレーム分の骨格座標を取得しました（ピクセル単位）")
            st.dataframe(df.head())
        else:
            st.warning("骨格情報が取得できませんでした。")

# ダウンロード＆ZIPまとめ
if st.session_state.csv_ready and not st.session_state.downloaded:
    st.markdown("---")

    # 一時CSV保存
    csv_path = os.path.join(tempfile.gettempdir(), "pose_output.csv")
    with open(csv_path, "wb") as f:
        f.write(st.session_state.csv_data)

    # ZIPファイル生成
    zip_path = os.path.join(tempfile.gettempdir(), "pose_output_bundle.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(csv_path, arcname="pose_output.csv")
        zipf.write(st.session_state.annotated_video_path, arcname="annotated_output.mp4")

    # ダウンロードボタン（ZIP）
    with open(zip_path, "rb") as zf:
        zip_bytes = zf.read()
        downloaded = st.download_button(
            "CSV（ピクセル座標）＋骨格動画をまとめてダウンロード (ZIP)",
            data=zip_bytes,
            file_name="pose_output_bundle.zip",
            mime="application/zip"
        )

    # 削除処理
    if downloaded:
        try:
            for path in [video_path, csv_path, zip_path, st.session_state.annotated_video_path]:
                if path and os.path.exists(path):
                    os.remove(path)
            st.success("一時ファイルをすべて削除しました")
        except Exception as e:
            st.warning(f"削除エラー: {e}")
        st.session_state.downloaded = True
        st.session_state.video_path = None
        st.session_state.csv_ready = False

# === 角度計算＆スティックピクチャ表示ボタン ===
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'MS Gothic'
if st.button("肩関節解析とスティックピクチャを表示"):
    st.info("CSVを読み込んで肩関節角度を解析＆スティックピクチャを描画します")

    # --- CSV 読み込み ---
    csv_path = os.path.join(tempfile.gettempdir(), "pose_output.csv")
    df = pd.read_csv(csv_path)
    n_frames = len(df)
    n_landmarks = 33

    pos_data_x = np.zeros((n_frames, n_landmarks))
    pos_data_y = np.zeros((n_frames, n_landmarks))
    for i in range(n_landmarks):
        pos_data_x[:, i] = df[f"x_{i}_px"].values
        pos_data_y[:, i] = -df[f"y_{i}_px"].values  # 上方向正に反転

    def calangle(v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.degrees(np.arccos(dot / (norm1 * norm2)))

    # --- 座標データ構築 ---
    mid_head = np.array([ (pos_data_x[:, 2] + pos_data_x[:, 5]) / 2,
                          (pos_data_y[:, 2] + pos_data_y[:, 5]) / 2 ]).T
    l_shoul = np.array([ pos_data_x[:, 11], pos_data_y[:, 11] ]).T
    l_elb   = np.array([ pos_data_x[:, 13], pos_data_y[:, 13] ]).T
    l_wri   = np.array([ pos_data_x[:, 15], pos_data_y[:, 15] ]).T
    r_shoul = np.array([ pos_data_x[:, 12], pos_data_y[:, 12] ]).T
    r_elb   = np.array([ pos_data_x[:, 14], pos_data_y[:, 14] ]).T
    r_wri   = np.array([ pos_data_x[:, 16], pos_data_y[:, 16] ]).T

    # --- 角度計算 ---
    vertical = np.array([0, 1])
    horizontal = np.array([1, 0])

    # 左肩
    vec_abd_l = l_elb - l_shoul
    ang_abd_l = 180 - np.array([calangle(v, vertical) for v in vec_abd_l])
    max_abd_l = np.max(ang_abd_l)
    tmg_abd_l = np.argmax(ang_abd_l)
    vec_elev_l = r_shoul - l_shoul
    ang_elev_l = 180 - np.array([calangle(v, horizontal) for v in vec_elev_l])
    max_elev_l = np.max(ang_elev_l)

    # 右肩
    vec_abd_r = r_elb - r_shoul
    ang_abd_r = 180 - np.array([calangle(v, vertical) for v in vec_abd_r])
    max_abd_r = np.max(ang_abd_r)
    tmg_abd_r = np.argmax(ang_abd_r)
    vec_elev_r = l_shoul - r_shoul
    ang_elev_r = np.array([calangle(v, horizontal) for v in vec_elev_r])
    max_elev_r = np.max(ang_elev_r)

    # --- 描画関数 ---
    def plot_stick(mid_head, l_shoul, r_shoul, elb, wri, side="L"):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(-300, 300)
        ax.set_ylim(-300, 300)
        
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.grid(alpha=0.2)

        # 点
        ax.scatter(mid_head[0], mid_head[1], color='magenta', s=300, alpha=0.8)
        ax.scatter(l_shoul[0], l_shoul[1], color='gray', s=100, alpha=0.8)
        ax.scatter(r_shoul[0], r_shoul[1], color='gray', s=100, alpha=0.8)
        ax.scatter(elb[0], elb[1], color='gray', s=100, alpha=0.8)
        ax.scatter(wri[0], wri[1], color='crimson', s=100, alpha=0.8)

        # 線
        ax.plot([l_shoul[0], r_shoul[0]], [l_shoul[1], r_shoul[1]], color='gray', linewidth=2, alpha=0.8)
        if side == "L":
            ax.plot([l_shoul[0], elb[0]], [l_shoul[1], elb[1]], color='gray', linewidth=2, alpha=0.8)
        else:
            ax.plot([r_shoul[0], elb[0]], [r_shoul[1], elb[1]], color='gray', linewidth=2, alpha=0.8)
        ax.plot([elb[0], wri[0]], [elb[1], wri[1]], color='gray', linewidth=2, alpha=0.8)

        return fig

    # --- 左上肢（最大外転） ---
    fig_l = plot_stick(
        mid_head[tmg_abd_l] - mid_head[0],
        l_shoul[tmg_abd_l] - mid_head[0],
        r_shoul[tmg_abd_l] - mid_head[0],
        l_elb[tmg_abd_l] - mid_head[0],
        l_wri[tmg_abd_l] - mid_head[0],
        side="L"
    )
    st.pyplot(fig_l)
    st.markdown(f"左肩最大外転角：**{max_abd_l:.1f}°**｜最大挙上角：**{max_elev_l:.1f}°**")

    # --- 右上肢（最大外転） ---
    fig_r = plot_stick(
        mid_head[tmg_abd_r] - mid_head[0],
        l_shoul[tmg_abd_r] - mid_head[0],
        r_shoul[tmg_abd_r] - mid_head[0],
        r_elb[tmg_abd_r] - mid_head[0],
        r_wri[tmg_abd_r] - mid_head[0],
        side="R"
    )
    st.pyplot(fig_r)
    st.markdown(f"右肩最大外転角：**{max_abd_r:.1f}°**｜最大挙上角：**{max_elev_r:.1f}°**")
