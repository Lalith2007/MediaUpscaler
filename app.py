from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import os, json, cv2, numpy as np, threading, uuid, requests, base64, mimetypes
from datetime import datetime
from PIL import Image

load_dotenv()
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'dev-key-12345'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///editing_suite.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")
oauth = OAuth(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


# Timezone helper
def to_ist(dt, fmt="%b %d, %Y at %I:%M %p"):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("Asia/Kolkata")).strftime(fmt)


app.jinja_env.filters["to_ist"] = to_ist


# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(200), unique=True)
    name = db.Column(db.String(200))
    email = db.Column(db.String(200), unique=True)
    profile_pic = db.Column(db.String(500))
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    country = db.Column(db.String(100))
    interests = db.Column(db.String(500))
    bio = db.Column(db.Text)
    is_profile_complete = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processing_queue = db.relationship('ProcessingJob', backref='user', lazy=True)


# Processing Job model
class ProcessingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_id = db.Column(db.String(100), unique=True)
    job_type = db.Column(db.String(50))
    input_file = db.Column(db.String(500))
    output_file = db.Column(db.String(500))
    status = db.Column(db.String(20), default='pending')
    progress = db.Column(db.Integer, default=0)
    settings = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.String(500))


google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def emit_job_progress(job_id, pct, message='', download_url=None):
    payload = {'job_id': job_id, 'progress': int(pct), 'message': message}
    if download_url:
        payload['download_url'] = download_url
    socketio.emit('upscale_progress', payload)


def upscale_image(input_path, output_path, scale=4, job_id=None, preset='balanced'):
    try:
        print(f"\n=== UPSCALE START ===")
        print(f"Job: {job_id} | Input: {input_path} | Output: {output_path} | Scale: {scale}x | Preset: {preset}")

        emit_job_progress(job_id, 3, 'Uploading to AI API...')

        with open(input_path, 'rb') as f:
            img_bytes = f.read()

        print(f"📤 Sending {len(img_bytes)} bytes to AI API")
        img_b64 = base64.b64encode(img_bytes).decode()
        payload = {'image': img_b64, 'scale': scale, 'job_id': job_id, 'preset': preset}

        response = requests.post('http://localhost:5001/upscale', json=payload, timeout=600)
        print(f"📥 Response Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result_b64 = data['upscaled_image']
                result_bytes = base64.b64decode(result_b64)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, 'wb') as f:
                    f.write(result_bytes)

                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✅ Output saved: {output_path} ({file_size} bytes)")
                else:
                    print(f"❌ Failed to save output to: {output_path}")
                    return False, "Failed to save output file"

                method = data.get('method', 'Enhanced LANCZOS')
                emit_job_progress(job_id, 100, f'Complete ({method})')
                return True, f"Upscaled {scale}x ({method})"

        print(f"❌ API returned status {response.status_code}")
        return False, f"API Error {response.status_code}"
    except Exception as e:
        print(f"❌ Exception in upscale_image: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def upscale_video(input_path, output_path, scale=2, job_id=None, preset='balanced'):
    try:
        with open(input_path, 'rb') as video_file:
            files = {'video': (os.path.basename(input_path), video_file, 'video/mp4')}  # Adjust mime type as necessary
            data = {'scale': scale, 'preset': preset, 'job_id': job_id}
            response = requests.post('http://localhost:5001/upscale-video', files=files, data=data, timeout=7200)  # long timeout for big videos

        if response.status_code == 200:
            json_data = response.json()
            video_b64 = json_data.get('video')
            if not video_b64:
                return False, "No video data returned from AI server"
            video_bytes = base64.b64decode(video_b64)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f_out:
                f_out.write(video_bytes)

            return True, "Video upscaled successfully with AI server"
        else:
            return False, f"AI server error {response.status_code}: {response.text}"

    except Exception as e:
        print(f"Exception forwarding video to AI server: {e}")
        return False, str(e)


def process_job(job_id, job_type, input_path, output_path, scale, preset):
    with app.app_context():
        job = ProcessingJob.query.filter_by(job_id=job_id).first()
        try:
            job.status = 'processing'
            job.progress = 1
            db.session.commit()

            if job_type == 'image':
                success, msg = upscale_image(input_path, output_path, scale, job_id, preset)
            else:
                success, msg = upscale_video(input_path, output_path, scale, job_id, preset)

            if success:
                job.status = 'completed'
                job.progress = 100
                job.completed_at = datetime.utcnow()
                job.error_message = None
                job.output_file = output_path
                db.session.commit()

                if os.path.exists(output_path):
                    print(f"✅ Job {job_id} completed - file verified at: {output_path}")
                else:
                    print(f"⚠️ Job {job_id} marked complete but file missing: {output_path}")

                dl_url = f'/api/download/{job_id}'

                emit_job_progress(job_id, 100, 'Complete', download_url=dl_url)
                socketio.emit('upscale_done', {'job_id': job_id, 'status': 'completed', 'download_url': dl_url})
            else:
                job.status = 'failed'
                job.progress = 0
                job.error_message = msg
                db.session.commit()
                print(f"❌ Job {job_id} failed: {msg}")
                socketio.emit('upscale_done', {'job_id': job_id, 'status': 'failed', 'error': msg})
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            db.session.commit()
            print(f"❌ Job {job_id} exception: {e}")
            import traceback
            traceback.print_exc()
            socketio.emit('upscale_done', {'job_id': job_id, 'status': 'failed', 'error': str(e)})


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login')
def login():
    return google.authorize_redirect(url_for('authorize', _external=True))


@app.route('/authorize')
def authorize():
    token = google.authorize_access_token()
    user_info = google.parse_id_token(token, nonce=None)
    user = User.query.filter_by(google_id=user_info['sub']).first()

    if not user:
        user = User(
            google_id=user_info['sub'],
            name=user_info.get('name'),
            email=user_info.get('email'),
            profile_pic=user_info.get('picture'),
            is_profile_complete=False
        )
        db.session.add(user)
        db.session.commit()

    login_user(user)

    if not user.is_profile_complete:
        return redirect(url_for('complete_profile'))

    return redirect(url_for('dashboard'))


@app.route('/complete-profile', methods=['GET', 'POST'])
@login_required
def complete_profile():
    if request.method == 'POST':
        current_user.first_name = request.form.get('first_name')
        current_user.last_name = request.form.get('last_name')
        current_user.age = int(request.form.get('age', 0)) if request.form.get('age') else None
        current_user.country = request.form.get('country')
        current_user.interests = request.form.get('interests')
        current_user.bio = request.form.get('bio')
        current_user.is_profile_complete = True
        db.session.commit()
        return redirect(url_for('dashboard'))

    return render_template('complete-profile.html')


@app.route('/dashboard')
@login_required
def dashboard():
    jobs = ProcessingJob.query.filter_by(user_id=current_user.id).order_by(ProcessingJob.created_at.desc()).limit(10).all()
    return render_template('dashboard.html', jobs=jobs)


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')


@app.route('/history')
@login_required
def history():
    return render_template('history.html')


@app.route('/image-upscaler')
@login_required
def image_upscaler():
    return render_template('image-upscaler.html')


@app.route('/video-upscaler')
@login_required
def video_upscaler():
    return render_template('video-upscaler.html')


@app.route('/frame-interpolator')
@login_required
def frame_interpolator():
    return render_template('frame-interpolator.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    job_type = request.form.get('type', 'image')
    scale = int(request.form.get('scale', 4))
    preset = request.form.get('preset', 'balanced')

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    job_id = str(uuid.uuid4())

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(input_path)

    print(f"📁 Saved upload: {input_path} ({os.path.getsize(input_path)} bytes)")

    ext = os.path.splitext(filename)[1] or '.png'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_upscaled{ext}")

    job = ProcessingJob(
        user_id=current_user.id,
        job_id=job_id,
        job_type=job_type,
        input_file=input_path,
        output_file=output_path,
        status='pending',
        progress=0,
        settings={'scale': scale, 'preset': preset}
    )
    db.session.add(job)
    db.session.commit()

    thread = threading.Thread(target=process_job, args=(job_id, job_type, input_path, output_path, scale, preset))
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'job_id': job_id,
        'status_url': url_for('api_job_status', job_id=job_id),
        'download_url': None
    })


@app.route('/api/job/<job_id>', methods=['GET'])
@login_required
def api_job_status(job_id):
    job = ProcessingJob.query.filter_by(job_id=job_id, user_id=current_user.id).first()
    if not job:
        return jsonify({'error': 'Not found'}), 404

    download_url = None
    if job.status == 'completed' and job.output_file and os.path.exists(job.output_file):
        download_url = f'/api/download/{job_id}'

    return jsonify({
        'job_id': job.job_id,
        'status': job.status,
        'progress': job.progress,
        'output_file': job.output_file if job.status == 'completed' else None,
        'download_url': download_url,
        'error': job.error_message,
        'settings': job.settings,
        'created_at': job.created_at.isoformat() + 'Z',
        'completed_at': job.completed_at.isoformat() + 'Z' if job.completed_at else None
    })


@app.route('/api/download/<job_id>')
@login_required
def api_download(job_id):
    try:
        job = ProcessingJob.query.filter_by(job_id=job_id, user_id=current_user.id).first()

        if not job:
            print(f"❌ Download: Job not found: {job_id}")
            return "Job not found", 404

        if job.status != 'completed':
            print(f"❌ Download: Job not completed: {job_id} (status: {job.status})")
            return f"Job not ready (status: {job.status})", 400

        if not job.output_file:
            print(f"❌ Download: No output_file in DB for job: {job_id}")
            return "Output file path not set", 500

        if not os.path.isabs(job.output_file):
            abs_output_path = os.path.join(os.getcwd(), job.output_file)
        else:
            abs_output_path = job.output_file

        if not os.path.exists(abs_output_path):
            print(f"❌ Download: File not found at: {abs_output_path}")
            print(f"   Working directory: {os.getcwd()}")
            print(f"   Stored path: {job.output_file}")
            return f"Output file not found on server", 404

        original_name = os.path.basename(job.input_file)
        name, ext = os.path.splitext(original_name)
        scale = job.settings.get('scale', 4) if job.settings else 4
        preset = job.settings.get('preset', 'balanced') if job.settings else 'balanced'
        download_name = f"{name}_upscaled_{scale}x_{preset}{ext}"

        ext_lower = ext.lower()
        if ext_lower in ['.png']:
            mime_type = 'image/png'
        elif ext_lower in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif ext_lower in ['.gif']:
            mime_type = 'image/gif'
        elif ext_lower in ['.webp']:
            mime_type = 'image/webp'
        elif ext_lower in ['.mp4']:
            mime_type = 'video/mp4'
        elif ext_lower in ['.mov']:
            mime_type = 'video/quicktime'
        else:
            mime_type = 'application/octet-stream'

        file_size = os.path.getsize(abs_output_path)
        print(f"\n📥 DOWNLOAD REQUEST")
        print(f"   Job ID: {job_id}")
        print(f"   Filename: {download_name}")
        print(f"   Path: {abs_output_path}")
        print(f"   Size: {file_size:,} bytes")
        print(f"   MIME: {mime_type}\n")

        response = send_file(
            abs_output_path,
            mimetype=mime_type,
            as_attachment=True,
            download_name=download_name
        )

        response.headers['Content-Type'] = mime_type
        response.headers['Content-Disposition'] = f'attachment; filename="{download_name}"'

        return response

    except Exception as e:
        print(f"❌ Download exception: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Download error: {str(e)}", 500


@app.route('/api/user-stats')
@login_required
def api_user_stats():
    total = ProcessingJob.query.filter_by(user_id=current_user.id).count()
    completed = ProcessingJob.query.filter_by(user_id=current_user.id, status='completed').count()
    failed = ProcessingJob.query.filter_by(user_id=current_user.id, status='failed').count()
    return jsonify({'total': total, 'completed': completed, 'failed': failed})


@app.route('/api/user-jobs')
@login_required
def api_user_jobs():
    jobs = ProcessingJob.query.filter_by(user_id=current_user.id).order_by(ProcessingJob.created_at.desc()).all()
    return jsonify([{
        'job_id': job.job_id,
        'job_type': job.job_type,
        'status': job.status,
        'progress': job.progress,
        'input_file': job.input_file,
        'output_file': job.output_file,
        'settings': job.settings,
        'created_at': job.created_at.isoformat(),
        'completed_at': job.completed_at.isoformat() if job.completed_at else None
    } for job in jobs])


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, host='localhost', port=8000)
