from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
import os
import threading
import time
import cv2
import base64
from process_video import process_video, LaneDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key_for_socketio'
socketio = SocketIO(app, cors_allowed_origins="*")

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

processing_status = {}
active_streams = {}

def allowed_file(filename):
    """Dosya uzantısını kontrol et"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_with_streaming(input_path, output_path, task_id):
    """Video işleme ve canlı streaming"""
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'message': 'Video işleniyor...',
            'progress': 0
        }
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Video açılamadı")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, first_frame = cap.read()
        if not ret:
            raise Exception("İlk frame okunamadı")
        
        detector = LaneDetector()
        first_frame = detector.resize_frame_if_needed(first_frame)
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        stream_fps = min(fps, 10)
        frame_skip = max(1, fps // stream_fps)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = detector.process_frame(frame)
            
            out.write(processed_frame)
            
            if frame_count % frame_skip == 0 and task_id in active_streams:
                try:
                    stream_frame = cv2.resize(processed_frame, (640, 360))
                    
                    _, buffer = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    socketio.emit('video_frame', {
                        'frame': frame_base64,
                        'progress': (frame_count / total_frames) * 100,
                        'frame_info': {
                            'current': frame_count,
                            'total': total_frames,
                            'fps': fps
                        }
                    }, room=task_id)
                    
                except Exception as e:
                    print(f"Streaming hatası: {e}")
            
            frame_count += 1
            
            progress = (frame_count / total_frames) * 100
            processing_status[task_id]['progress'] = progress
            
            if task_id not in processing_status:
                break
        
        cap.release()
        out.release()
        
        if task_id in active_streams:
            socketio.emit('video_complete', {'message': 'Video işleme tamamlandı!'}, room=task_id)
            del active_streams[task_id]
        
        processing_status[task_id] = {
            'status': 'completed',
            'message': 'Video başarıyla işlendi!',
            'output_file': os.path.basename(output_path),
            'progress': 100
        }
        
    except Exception as e:
        processing_status[task_id] = {
            'status': 'error',
            'message': f'Hata: {str(e)}',
            'progress': 0
        }
        
        if task_id in active_streams:
            socketio.emit('video_error', {'message': str(e)}, room=task_id)
            del active_streams[task_id]

@app.route('/upload', methods=['POST'])
def upload_video():
    """Video yükleme endpoint'i"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': 'Video dosyası bulunamadı'})
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Dosya seçilmedi'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Desteklenmeyen dosya formatı'})
        
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        input_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        
        file.save(input_path)
        
        output_filename = f"processed_{timestamp}_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        task_id = f"task_{timestamp}"
        
        thread = threading.Thread(
            target=process_video_with_streaming,
            args=(input_path, output_path, task_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Video yüklendi, işleme başlatıldı',
            'task_id': task_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'})

@app.route('/stop/<task_id>', methods=['POST'])
def stop_processing(task_id):
    """İşlemi durdur"""
    if task_id in processing_status:
        del processing_status[task_id]
    if task_id in active_streams:
        del active_streams[task_id]
    return jsonify({'success': True, 'message': 'İşlem durduruldu'})

@app.route('/status/<task_id>')
def get_status(task_id):
    """İşlem durumu kontrolü"""
    if task_id in processing_status:
        return jsonify(processing_status[task_id])
    else:
        return jsonify({
            'status': 'not_found',
            'message': 'Task bulunamadı',
            'progress': 0
        })

@app.route('/download/<filename>')
def download_file(filename):
    """İşlenmiş video indirme"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename);
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Dosya bulunamadı'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
        
    except Exception as e:
        return jsonify({'error': f'İndirme hatası: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Sistem durumu kontrolü"""
    return jsonify({
        'status': 'healthy',
        'upload_folder': os.path.exists(UPLOAD_FOLDER),
        'output_folder': os.path.exists(OUTPUT_FOLDER),
        'active_tasks': len(processing_status)
    })

@app.route('/')
def index():
    """Ana sayfa - Modern UI"""
    return '''
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <title>Şerit Takip Sistemi</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                --bg-primary: #0a0a0f;
                --bg-secondary: #12121a;
                --bg-card: #1a1a24;
                --bg-card-hover: #22222e;
                --accent-primary: #6366f1;
                --accent-secondary: #8b5cf6;
                --accent-gradient: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --text-muted: #64748b;
                --border-color: rgba(99, 102, 241, 0.2);
                --glow: 0 0 40px rgba(99, 102, 241, 0.3);
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                min-height: 100vh;
                overflow-x: hidden;
            }
            
            /* Animated Background */
            .bg-animation {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
                overflow: hidden;
            }
            
            .bg-animation::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
                            radial-gradient(circle at 40% 40%, rgba(168, 85, 247, 0.05) 0%, transparent 40%);
                animation: bgPulse 15s ease-in-out infinite;
            }
            
            @keyframes bgPulse {
                0%, 100% { transform: translate(0, 0) rotate(0deg); }
                33% { transform: translate(2%, 2%) rotate(1deg); }
                66% { transform: translate(-1%, 1%) rotate(-1deg); }
            }
            
            /* Header */
            .header {
                padding: 24px 40px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 1px solid var(--border-color);
                backdrop-filter: blur(20px);
                position: sticky;
                top: 0;
                z-index: 100;
            }
            
            .logo {
                display: flex;
                align-items: center;
                gap: 14px;
            }
            
            .logo-icon {
                width: 48px;
                height: 48px;
                background: var(--accent-gradient);
                border-radius: 14px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 22px;
                box-shadow: var(--glow);
            }
            
            .logo-text {
                font-size: 22px;
                font-weight: 700;
                background: var(--accent-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .logo-subtitle {
                font-size: 12px;
                color: var(--text-muted);
                font-weight: 400;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            
            .header-status {
                display: flex;
                align-items: center;
                gap: 20px;
            }
            
            .status-badge {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 10px 18px;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 30px;
                font-size: 13px;
                font-weight: 500;
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                background: var(--success);
                border-radius: 50%;
                animation: pulse 2s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.6; transform: scale(1.1); }
            }
            
            /* Main Container */
            .main-container {
                max-width: 1440px;
                margin: 0 auto;
                padding: 40px;
                display: grid;
                grid-template-columns: 420px 1fr;
                gap: 32px;
            }
            
            /* Cards */
            .card {
                background: var(--bg-card);
                border-radius: 24px;
                border: 1px solid var(--border-color);
                overflow: hidden;
                transition: all 0.3s ease;
            }
            
            .card:hover {
                border-color: rgba(99, 102, 241, 0.4);
                box-shadow: var(--glow);
            }
            
            .card-header {
                padding: 24px 28px;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .card-title {
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 16px;
                font-weight: 600;
            }
            
            .card-title i {
                width: 36px;
                height: 36px;
                background: var(--accent-gradient);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
            }
            
            .card-body {
                padding: 28px;
            }
            
            /* Upload Zone */
            .upload-zone {
                border: 2px dashed var(--border-color);
                border-radius: 20px;
                padding: 48px 32px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.03), rgba(139, 92, 246, 0.03));
            }
            
            .upload-zone:hover, .upload-zone.dragover {
                border-color: var(--accent-primary);
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
                transform: scale(1.01);
            }
            
            .upload-icon {
                width: 80px;
                height: 80px;
                background: var(--accent-gradient);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 24px;
                font-size: 32px;
                box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
            }
            
            .upload-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            .upload-subtitle {
                color: var(--text-secondary);
                font-size: 14px;
                margin-bottom: 20px;
            }
            
            .upload-formats {
                display: flex;
                justify-content: center;
                gap: 10px;
                flex-wrap: wrap;
            }
            
            .format-badge {
                padding: 6px 14px;
                background: rgba(99, 102, 241, 0.1);
                border-radius: 8px;
                font-size: 12px;
                font-weight: 500;
                color: var(--accent-primary);
            }
            
            .file-input {
                display: none;
            }
            
            /* Selected File Display */
            .selected-file {
                display: none;
                align-items: center;
                gap: 16px;
                padding: 20px;
                background: rgba(99, 102, 241, 0.08);
                border: 1px solid var(--accent-primary);
                border-radius: 16px;
                margin-top: 20px;
            }
            
            .selected-file.show {
                display: flex;
            }
            
            .file-icon {
                width: 48px;
                height: 48px;
                background: var(--accent-gradient);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
            }
            
            .file-info {
                flex: 1;
            }
            
            .file-name {
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 4px;
            }
            
            .file-size {
                font-size: 12px;
                color: var(--text-muted);
            }
            
            .file-remove {
                width: 32px;
                height: 32px;
                background: rgba(239, 68, 68, 0.1);
                border: none;
                border-radius: 8px;
                color: var(--danger);
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .file-remove:hover {
                background: rgba(239, 68, 68, 0.2);
            }
            
            /* Buttons */
            .btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                padding: 16px 32px;
                border: none;
                border-radius: 14px;
                font-family: inherit;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
            }
            
            .btn-primary {
                background: var(--accent-gradient);
                color: white;
                box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4);
            }
            
            .btn-primary:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 12px 40px rgba(99, 102, 241, 0.5);
            }
            
            .btn-primary:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            
            .btn-danger {
                background: linear-gradient(135deg, #ef4444, #dc2626);
                color: white;
            }
            
            .btn-success {
                background: linear-gradient(135deg, #10b981, #059669);
                color: white;
            }
            
            .btn-group {
                display: flex;
                gap: 12px;
                margin-top: 20px;
            }
            
            .btn-group .btn {
                flex: 1;
            }
            
            /* Progress Section */
            .progress-section {
                display: none;
                margin-top: 28px;
            }
            
            .progress-section.show {
                display: block;
            }
            
            .progress-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 14px;
            }
            
            .progress-label {
                font-size: 14px;
                font-weight: 500;
            }
            
            .progress-value {
                font-size: 14px;
                font-weight: 700;
                color: var(--accent-primary);
            }
            
            .progress-bar {
                height: 10px;
                background: rgba(99, 102, 241, 0.15);
                border-radius: 10px;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background: var(--accent-gradient);
                border-radius: 10px;
                width: 0%;
                transition: width 0.4s ease;
                position: relative;
            }
            
            .progress-fill::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                animation: shimmer 1.5s infinite;
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            
            /* Stats Grid */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 14px;
                margin-top: 24px;
            }
            
            .stat-card {
                background: var(--bg-secondary);
                border-radius: 14px;
                padding: 18px;
                text-align: center;
            }
            
            .stat-value {
                font-size: 24px;
                font-weight: 700;
                background: var(--accent-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .stat-label {
                font-size: 12px;
                color: var(--text-muted);
                margin-top: 4px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            /* Status Messages */
            .status-message {
                display: none;
                align-items: center;
                gap: 14px;
                padding: 18px 22px;
                border-radius: 14px;
                margin-top: 24px;
                font-size: 14px;
                font-weight: 500;
            }
            
            .status-message.show {
                display: flex;
            }
            
            .status-message.info {
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.3);
                color: var(--accent-primary);
            }
            
            .status-message.success {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: var(--success);
            }
            
            .status-message.error {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: var(--danger);
            }
            
            .status-message.warning {
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: var(--warning);
            }
            
            /* Video Section */
            .video-card {
                display: flex;
                flex-direction: column;
            }
            
            .video-container {
                position: relative;
                background: var(--bg-secondary);
                border-radius: 20px;
                overflow: hidden;
                aspect-ratio: 16 / 9;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            #videoPlayer {
                width: 100%;
                height: 100%;
                object-fit: contain;
            }
            
            .video-placeholder {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.05));
            }
            
            .video-placeholder.hidden {
                display: none;
            }
            
            .placeholder-icon {
                width: 100px;
                height: 100px;
                background: rgba(99, 102, 241, 0.1);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 40px;
                color: var(--text-muted);
                margin-bottom: 20px;
            }
            
            .placeholder-text {
                color: var(--text-muted);
                font-size: 16px;
            }
            
            /* Video Overlay Info */
            .video-overlay {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                padding: 24px;
                background: linear-gradient(transparent, rgba(0,0,0,0.9));
                display: none;
            }
            
            .video-overlay.show {
                display: block;
            }
            
            .overlay-stats {
                display: flex;
                gap: 24px;
            }
            
            .overlay-stat {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .overlay-stat i {
                width: 32px;
                height: 32px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
            }
            
            .overlay-stat-value {
                font-weight: 600;
                font-size: 14px;
            }
            
            .overlay-stat-label {
                font-size: 11px;
                color: var(--text-muted);
            }
            
            /* Download Section */
            .download-section {
                display: none;
                margin-top: 24px;
            }
            
            .download-section.show {
                display: block;
            }
            
            /* Processing Animation */
            .processing-indicator {
                display: none;
                align-items: center;
                justify-content: center;
                gap: 12px;
                padding: 16px;
                background: rgba(99, 102, 241, 0.08);
                border-radius: 12px;
                margin-top: 20px;
            }
            
            .processing-indicator.show {
                display: flex;
            }
            
            .spinner {
                width: 24px;
                height: 24px;
                border: 3px solid rgba(99, 102, 241, 0.2);
                border-top-color: var(--accent-primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            /* Responsive */
            @media (max-width: 1200px) {
                .main-container {
                    grid-template-columns: 1fr;
                    padding: 24px;
                }
            }
            
            @media (max-width: 640px) {
                .header {
                    padding: 16px 20px;
                    flex-direction: column;
                    gap: 16px;
                }
                
                .main-container {
                    padding: 16px;
                }
                
                .card-body {
                    padding: 20px;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                
                .btn-group {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <div class="bg-animation"></div>
        
        <header class="header">
            <div class="logo">
                <div class="logo-icon">
                    <i class="fas fa-road"></i>
                </div>
                <div>
                    <div class="logo-text">LaneVision AI</div>
                    <div class="logo-subtitle">Akıllı Şerit Takip Sistemi</div>
                </div>
            </div>
            <div class="header-status">
                <div class="status-badge">
                    <div class="status-dot"></div>
                    Sistem Aktif
                </div>
            </div>
        </header>
        
        <main class="main-container">
            <!-- Upload Section -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-cloud-upload-alt"></i>
                        Video Yükle
                    </div>
                </div>
                <div class="card-body">
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="upload-zone" id="uploadZone">
                            <div class="upload-icon">
                                <i class="fas fa-film"></i>
                            </div>
                            <div class="upload-title">Video dosyanızı sürükleyip bırakın</div>
                            <div class="upload-subtitle">veya seçmek için tıklayın</div>
                            <div class="upload-formats">
                                <span class="format-badge">MP4</span>
                                <span class="format-badge">AVI</span>
                                <span class="format-badge">MOV</span>
                                <span class="format-badge">MKV</span>
                            </div>
                            <input type="file" name="video" id="fileInput" class="file-input" accept=".mp4,.avi,.mov,.mkv" required>
                        </div>
                        
                        <div class="selected-file" id="selectedFile">
                            <div class="file-icon">
                                <i class="fas fa-video"></i>
                            </div>
                            <div class="file-info">
                                <div class="file-name" id="fileName"></div>
                                <div class="file-size" id="fileSize"></div>
                            </div>
                            <button type="button" class="file-remove" id="fileRemove">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                        
                        <button type="submit" class="btn btn-primary" id="uploadBtn" style="margin-top: 24px;">
                            <i class="fas fa-play"></i>
                            İşlemeyi Başlat
                        </button>
                    </form>
                    
                    <!-- Progress Section -->
                    <div class="progress-section" id="progressSection">
                        <div class="progress-header">
                            <span class="progress-label">İşleniyor...</span>
                            <span class="progress-value" id="progressValue">0%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressBar"></div>
                        </div>
                        
                        <div class="stats-grid" id="statsGrid">
                            <div class="stat-card">
                                <div class="stat-value" id="currentFrame">0</div>
                                <div class="stat-label">Frame</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="totalFrames">0</div>
                                <div class="stat-label">Toplam</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="processingFps">0</div>
                                <div class="stat-label">FPS</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="elapsedTime">0s</div>
                                <div class="stat-label">Süre</div>
                            </div>
                        </div>
                        
                        <div class="btn-group">
                            <button class="btn btn-danger" id="stopBtn">
                                <i class="fas fa-stop"></i>
                                Durdur
                            </button>
                        </div>
                    </div>
                    
                    <!-- Download Section -->
                    <div class="download-section" id="downloadSection">
                        <button class="btn btn-success" id="downloadBtn">
                            <i class="fas fa-download"></i>
                            Videoyu İndir
                        </button>
                    </div>
                    
                    <!-- Status Messages -->
                    <div class="status-message" id="statusMessage">
                        <i class="fas fa-info-circle" id="statusIcon"></i>
                        <span id="statusText"></span>
                    </div>
                    
                    <div class="processing-indicator" id="processingIndicator">
                        <div class="spinner"></div>
                        <span>Video yükleniyor...</span>
                    </div>
                </div>
            </div>
            
            <!-- Video Section -->
            <div class="card video-card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fas fa-tv"></i>
                        Canlı Önizleme
                    </div>
                </div>
                <div class="card-body">
                    <div class="video-container">
                        <img id="videoPlayer" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" alt="Video">
                        
                        <div class="video-placeholder" id="videoPlaceholder">
                            <div class="placeholder-icon">
                                <i class="fas fa-photo-video"></i>
                            </div>
                            <div class="placeholder-text">Video işlenmeye başladığında burada görünecek</div>
                        </div>
                        
                        <div class="video-overlay" id="videoOverlay">
                            <div class="overlay-stats">
                                <div class="overlay-stat">
                                    <i class="fas fa-images"></i>
                                    <div>
                                        <div class="overlay-stat-value" id="overlayFrame">0 / 0</div>
                                        <div class="overlay-stat-label">Frame</div>
                                    </div>
                                </div>
                                <div class="overlay-stat">
                                    <i class="fas fa-tachometer-alt"></i>
                                    <div>
                                        <div class="overlay-stat-value" id="overlayFps">0 FPS</div>
                                        <div class="overlay-stat-label">İşleme Hızı</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <script>
            const socket = io();
            let currentTaskId = null;
            let isProcessing = false;
            let startTime = null;

            const uploadForm = document.getElementById('uploadForm');
            const uploadZone = document.getElementById('uploadZone');
            const fileInput = document.getElementById('fileInput');
            const selectedFile = document.getElementById('selectedFile');
            const fileName = document.getElementById('fileName');
            const fileSize = document.getElementById('fileSize');
            const fileRemove = document.getElementById('fileRemove');
            const uploadBtn = document.getElementById('uploadBtn');
            const progressSection = document.getElementById('progressSection');
            const progressBar = document.getElementById('progressBar');
            const progressValue = document.getElementById('progressValue');
            const progressLabel = document.querySelector('.progress-label');
            const downloadSection = document.getElementById('downloadSection');
            const downloadBtn = document.getElementById('downloadBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusMessage = document.getElementById('statusMessage');
            const statusIcon = document.getElementById('statusIcon');
            const statusText = document.getElementById('statusText');
            const processingIndicator = document.getElementById('processingIndicator');
            const videoPlayer = document.getElementById('videoPlayer');
            const videoPlaceholder = document.getElementById('videoPlaceholder');
            const videoOverlay = document.getElementById('videoOverlay');

            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }

            uploadZone.addEventListener('click', () => fileInput.click());
            
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });
            
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });
            
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    handleFileSelect(files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                }
            });

            function handleFileSelect(file) {
                fileName.textContent = file.name;
                fileSize.textContent = formatFileSize(file.size);
                selectedFile.classList.add('show');
                uploadZone.style.display = 'none';
            }

            fileRemove.addEventListener('click', () => {
                fileInput.value = '';
                selectedFile.classList.remove('show');
                uploadZone.style.display = 'block';
            });

            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                if (isProcessing) {
                    showStatus('Zaten bir işlem devam ediyor!', 'warning');
                    return;
                }
                
                const formData = new FormData(uploadForm);
                
                uploadBtn.disabled = true;
                processingIndicator.classList.add('show');
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    processingIndicator.classList.remove('show');
                    
                    if (data.success) {
                        currentTaskId = data.task_id;
                        isProcessing = true;
                        startTime = Date.now();
                        
                        socket.emit('join', { task_id: currentTaskId });
                        
                        showStatus('Video işleniyor...', 'info');
                        progressSection.classList.add('show');
                        videoPlaceholder.classList.add('hidden');
                        videoOverlay.classList.add('show');
                        
                        checkStatus();
                    } else {
                        showStatus(data.message, 'error');
                        resetUpload();
                    }
                } catch (error) {
                    processingIndicator.classList.remove('show');
                    showStatus('Bağlantı hatası: ' + error.message, 'error');
                    resetUpload();
                }
            });

            stopBtn.addEventListener('click', async () => {
                if (currentTaskId && isProcessing) {
                    const response = await fetch(`/stop/${currentTaskId}`, { method: 'POST' });
                    const data = await response.json();
                    showStatus('İşlem durduruldu', 'warning');
                    resetInterface();
                }
            });

            downloadBtn.addEventListener('click', async () => {
                if (currentTaskId) {
                    const response = await fetch(`/status/${currentTaskId}`);
                    const data = await response.json();
                    if (data.output_file) {
                        window.location.href = `/download/${data.output_file}`;
                    }
                }
            });

            socket.on('video_frame', (data) => {
                videoPlayer.src = 'data:image/jpeg;base64,' + data.frame;
                updateProgress(data.progress);
                
                if (data.frame_info) {
                    document.getElementById('currentFrame').textContent = data.frame_info.current;
                    document.getElementById('totalFrames').textContent = data.frame_info.total;
                    document.getElementById('processingFps').textContent = data.frame_info.fps;
                    document.getElementById('overlayFrame').textContent = `${data.frame_info.current} / ${data.frame_info.total}`;
                    document.getElementById('overlayFps').textContent = `${data.frame_info.fps} FPS`;
                    
                    if (startTime) {
                        const elapsed = Math.round((Date.now() - startTime) / 1000);
                        document.getElementById('elapsedTime').textContent = elapsed + 's';
                    }
                }
            });

            socket.on('video_complete', (data) => {
                showStatus('Video başarıyla işlendi!', 'success');
                updateProgress(100);
                progressLabel.textContent = "Tamamlandı";
                downloadSection.classList.add('show');
                isProcessing = false;
                resetUpload();
            });
            
            socket.on('video_error', (data) => {
                showStatus('Hata: ' + data.message, 'error');
                resetInterface();
            });

            function showStatus(message, type) {
                statusMessage.className = `status-message show ${type}`;
                statusText.textContent = message;
                
                const icons = {
                    info: 'fa-info-circle',
                    success: 'fa-check-circle',
                    error: 'fa-exclamation-circle',
                    warning: 'fa-exclamation-triangle'
                };
                statusIcon.className = `fas ${icons[type]}`;
            }

            function updateProgress(percent) {
                progressBar.style.width = percent + '%';
                progressValue.textContent = Math.round(percent) + '%';
            }

            function resetUpload() {
                uploadBtn.disabled = false;
            }

            function resetInterface() {
                isProcessing = false;
                currentTaskId = null;
                startTime = null;
                resetUpload();
                progressSection.classList.remove('show');
                downloadSection.classList.remove('show');
                videoOverlay.classList.remove('show');
                videoPlaceholder.classList.remove('hidden');
                selectedFile.classList.remove('show');
                uploadZone.style.display = 'block';
                fileInput.value = '';
            }

            async function checkStatus() {
                if (!currentTaskId || !isProcessing) return;
                
                try {
                    const response = await fetch(`/status/${currentTaskId}`);
                    const data = await response.json();
                    
                    if (data.status === 'completed') {
                        showStatus('Video başarıyla işlendi!', 'success');
                        updateProgress(100);
                        downloadSection.classList.add('show');
                        isProcessing = false;
                        resetUpload();
                    } else if (data.status === 'error') {
                        showStatus(data.message, 'error');
                        resetInterface();
                    } else if (data.status === 'processing') {
                        updateProgress(data.progress || 0);
                        setTimeout(checkStatus, 2000);
                    }
                } catch (error) {
                    console.error('Status check error:', error);
                    setTimeout(checkStatus, 2000);
                }
            }
        </script>
    </body>
    </html>
    '''


def cleanup_old_files():
    """24 saatten eski dosyaları temizle"""
    import glob
    current_time = time.time()
    
    for file_path in glob.glob(os.path.join(UPLOAD_FOLDER, "*")):
        if os.path.getmtime(file_path) < current_time - 24 * 3600:
            try:
                os.remove(file_path)
                print(f"Eski dosya silindi: {file_path}")
            except Exception as e:
                print(f"Dosya silinirken hata: {e}")
    
    for file_path in glob.glob(os.path.join(OUTPUT_FOLDER, "*")):
        if os.path.getmtime(file_path) < current_time - 24 * 3600:
            try:
                os.remove(file_path)
                print(f"Eski dosya silindi: {file_path}")
            except Exception as e:
                print(f"Dosya silinirken hata: {e}")

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'message': 'Dosya çok büyük (Max: 100MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint bulunamadı'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Sunucu hatası'}), 500

@socketio.on('join')
def on_join(data):
    """Client'ı task room'una ekle"""
    task_id = data['task_id']
    join_room(task_id)
    active_streams[task_id] = True
    emit('joined', {'task_id': task_id})

@socketio.on('disconnect')
def on_disconnect():
    """Client bağlantısı kesildiğinde"""
    print('Client disconnected')

if __name__ == '__main__':
    print("Şerit Tespit ve Canlı Görüntüleme Servisi Başlatılıyor...")
    print("Klasörler kontrol ediliyor...")
    print(f"   Upload: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"   Output: {os.path.abspath(OUTPUT_FOLDER)}")
    print("Server başlatılıyor: http://localhost:5000")
    print("Kullanım: Tarayıcıdan video yükleyip canlı şerit tespit analizi izleyin")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)