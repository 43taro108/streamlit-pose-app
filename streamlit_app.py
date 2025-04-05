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
import numpy as np

# 初期設定
st.set_page_config(page_title="MediaPipe CSV出力", layout="centered")
st.title("MediaPipe Pose × CSV出力")

st.info("10秒以内の動画のみ対応。アップロード後、自動処理＆CSV出力。動画や画像は保存されません。")

# セッションステート初期化
for key in ["video_path", "csv_ready", "csv_data"]:
    if key not in st.session_state:
        st.session_state[key] = None if "path" in key or "data" in key else False

# アップロード受付
uploaded_file = st.file_uploader("動画ファイルをアップロード（.mp4 / .mov / .avi）", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.session_state.video_path = tfile.name
    st.session_state.csv_ready = False

video_path = st.session_state.video_path

if video_path and os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    st.markdown(f"フレーム数: {frame_count}｜FPS: {fps:.2f}｜サイズ: {width}×{height}｜長さ: {duration:.1f} 秒")

    if duration > 10:
        st.warning("このアプリは10秒以内の動画に限定されています。短くトリミングしてください。")
    else:
        if st.button("骨格検出＋CSV出力"):
            with st.spinner("骨格検出とCSV出力中..."):
                mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
                all_data = []

                frame_idx = 0
                while cap.isOpened():
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

                    frame_idx += 1
                    del frame  # メモリ節約

                cap.release()
                pose.close()

            if all_data:
                header = ["frame"] + [f"x_{i}_px" for i in range(33)] + [f"y_{i}_px" for i in range(33)]
                csv_data = []
                for row in all_data:
                    # xとyを交互ではなく分離して保存
                    f_id = row[0]
                    x_values = row[1::2]
                    y_values = row[2::2]
                    csv_data.append([f_id] + x_values + y_values)

                df = pd.DataFrame(csv_data, columns=header)
                output_bytes = df.to_csv(index=False).encode("utf-8")
                del df  # メモリ解放

                st.download_button("CSVファイルをダウンロード", output_bytes, file_name="pose_output.csv", mime="text/csv")
                st.success("CSV出力が完了しました。ダウンロードしてご確認ください。")
            else:
                st.warning("骨格情報が検出されませんでした。カメラ映像が暗すぎる可能性があります。")

    try:
        os.remove(video_path)
        st.session_state.video_path = None
    except Exception as e:
        st.warning(f"一時ファイル削除エラー: {e}")
