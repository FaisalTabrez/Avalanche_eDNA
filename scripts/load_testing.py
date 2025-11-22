"""
Load testing and performance benchmarking
Using Locust for distributed load testing
"""
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import random
import json
import time
from faker import Faker

fake = Faker()


# ============================================================================
# User Behavior Classes
# ============================================================================

class AvalancheUser(HttpUser):
    """
    Base user class for Avalanche platform
    Simulates typical user behavior patterns
    """
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    
    def on_start(self):
        """Called when a user starts - perform login"""
        self.login()
    
    def login(self):
        """Authenticate user"""
        response = self.client.post("/api/auth/login", json={
            "username": f"testuser_{random.randint(1, 100)}",
            "password": "testpass123"
        })
        
        if response.status_code == 200:
            self.token = response.json().get("token")
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(3)
    def view_dashboard(self):
        """View main dashboard (common operation)"""
        self.client.get("/")
    
    @task(2)
    def list_datasets(self):
        """List user's datasets"""
        self.client.get("/api/datasets")
    
    @task(2)
    def list_analysis_runs(self):
        """List user's analysis runs"""
        self.client.get("/api/analysis-runs")
    
    @task(1)
    def view_dataset_details(self):
        """View specific dataset details"""
        dataset_id = random.randint(1, 100)
        self.client.get(f"/api/datasets/{dataset_id}")
    
    @task(1)
    def view_analysis_details(self):
        """View specific analysis run details"""
        run_id = random.randint(1, 100)
        self.client.get(f"/api/analysis-runs/{run_id}")
    
    @task(1)
    def search_taxonomy(self):
        """Search taxonomy predictions"""
        params = {
            "query": fake.word(),
            "confidence_min": 0.7,
            "page": 1,
            "per_page": 20
        }
        self.client.get("/api/taxonomy/search", params=params)


class UploadUser(HttpUser):
    """
    User focused on uploading and processing data
    Simulates data submission workflow
    """
    wait_time = between(5, 15)  # Uploads take longer
    
    def on_start(self):
        """Login"""
        self.login()
    
    def login(self):
        """Authenticate"""
        response = self.client.post("/api/auth/login", json={
            "username": f"upload_user_{random.randint(1, 50)}",
            "password": "testpass123"
        })
        
        if response.status_code == 200:
            self.token = response.json().get("token")
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(5)
    def upload_fasta(self):
        """Upload FASTA file"""
        # Generate fake FASTA content
        sequences = []
        for i in range(100):
            header = f">sequence_{i}_{fake.uuid4()}"
            seq = ''.join(random.choices('ATCG', k=500))
            sequences.append(f"{header}\n{seq}")
        
        fasta_content = "\n".join(sequences)
        
        files = {
            'file': ('test_sequences.fasta', fasta_content, 'text/plain')
        }
        data = {
            'dataset_name': f"Test Dataset {fake.uuid4()}",
            'description': fake.text(max_nb_chars=200)
        }
        
        self.client.post("/api/datasets/upload", files=files, data=data)
    
    @task(3)
    def submit_analysis(self):
        """Submit analysis job"""
        data = {
            'dataset_id': random.randint(1, 100),
            'analysis_type': random.choice(['taxonomy', 'novelty', 'clustering']),
            'parameters': {
                'min_confidence': 0.7,
                'reference_db': 'silva'
            }
        }
        
        self.client.post("/api/analysis-runs", json=data)
    
    @task(1)
    def check_job_status(self):
        """Check analysis job status"""
        run_id = random.randint(1, 100)
        self.client.get(f"/api/analysis-runs/{run_id}/status")


class ReportUser(HttpUser):
    """
    User focused on viewing and downloading reports
    Simulates report generation and export workflows
    """
    wait_time = between(2, 8)
    
    def on_start(self):
        """Login"""
        self.login()
    
    def login(self):
        """Authenticate"""
        response = self.client.post("/api/auth/login", json={
            "username": f"report_user_{random.randint(1, 50)}",
            "password": "testpass123"
        })
        
        if response.status_code == 200:
            self.token = response.json().get("token")
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(4)
    def list_reports(self):
        """List available reports"""
        self.client.get("/api/reports")
    
    @task(3)
    def view_report(self):
        """View report details"""
        report_id = random.randint(1, 100)
        self.client.get(f"/api/reports/{report_id}")
    
    @task(2)
    def generate_report(self):
        """Request report generation"""
        data = {
            'analysis_run_id': random.randint(1, 100),
            'report_type': random.choice(['summary', 'detailed', 'comparison']),
            'format': random.choice(['html', 'pdf', 'json'])
        }
        
        self.client.post("/api/reports/generate", json=data)
    
    @task(1)
    def download_report(self):
        """Download generated report"""
        report_id = random.randint(1, 100)
        self.client.get(f"/api/reports/{report_id}/download")
    
    @task(1)
    def export_data(self):
        """Export analysis data"""
        data = {
            'analysis_run_id': random.randint(1, 100),
            'format': random.choice(['csv', 'json', 'xlsx'])
        }
        
        self.client.post("/api/export", json=data)


# ============================================================================
# Event Handlers for Metrics
# ============================================================================

request_times = []
error_count = 0
success_count = 0


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track request metrics"""
    global request_times, error_count, success_count
    
    request_times.append(response_time)
    
    if exception:
        error_count += 1
    else:
        success_count += 1


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("=" * 80)
    print("Load Test Starting")
    print("=" * 80)
    
    if isinstance(environment.runner, MasterRunner):
        print("Running in distributed mode (master)")
    else:
        print("Running in standalone mode")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - print summary statistics"""
    global request_times, error_count, success_count
    
    print("\n" + "=" * 80)
    print("Load Test Summary")
    print("=" * 80)
    
    if request_times:
        avg_response_time = sum(request_times) / len(request_times)
        min_response_time = min(request_times)
        max_response_time = max(request_times)
        
        # Calculate percentiles
        sorted_times = sorted(request_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        print(f"\nRequests:")
        print(f"  Total: {len(request_times)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {error_count}")
        print(f"  Error Rate: {(error_count / len(request_times) * 100):.2f}%")
        
        print(f"\nResponse Times (ms):")
        print(f"  Average: {avg_response_time:.2f}")
        print(f"  Min: {min_response_time:.2f}")
        print(f"  Max: {max_response_time:.2f}")
        print(f"  50th percentile: {p50:.2f}")
        print(f"  95th percentile: {p95:.2f}")
        print(f"  99th percentile: {p99:.2f}")
    
    print("\n" + "=" * 80)


# ============================================================================
# Custom Load Shapes
# ============================================================================

from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    Step load pattern: gradually increase load in steps
    """
    step_time = 60  # Each step lasts 60 seconds
    step_load = 10  # Add 10 users per step
    spawn_rate = 2  # Spawn 2 users per second
    time_limit = 600  # Total test duration: 10 minutes
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        user_count = int((current_step + 1) * self.step_load)
        
        return (user_count, self.spawn_rate)


class SpikeLoadShape(LoadTestShape):
    """
    Spike load pattern: sudden traffic spike then return to baseline
    """
    baseline_users = 20
    spike_users = 200
    spike_start = 120  # Spike starts at 2 minutes
    spike_duration = 60  # Spike lasts 1 minute
    spawn_rate = 10
    time_limit = 300  # 5 minutes total
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        if self.spike_start <= run_time < (self.spike_start + self.spike_duration):
            return (self.spike_users, self.spawn_rate)
        else:
            return (self.baseline_users, self.spawn_rate)


class DoubleWaveLoadShape(LoadTestShape):
    """
    Double wave pattern: two peaks (simulating business hours)
    """
    time_limit = 600  # 10 minutes
    spawn_rate = 5
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        # Calculate user count based on sine wave
        import math
        phase = (run_time / self.time_limit) * 4 * math.pi
        user_count = int(50 + 50 * math.sin(phase))
        
        return (max(10, user_count), self.spawn_rate)


# ============================================================================
# Helper Functions
# ============================================================================

def run_load_test(host, users=100, spawn_rate=10, duration=60):
    """
    Run load test programmatically
    
    Args:
        host: Target host URL
        users: Number of concurrent users
        spawn_rate: Users spawned per second
        duration: Test duration in seconds
    """
    import subprocess
    
    cmd = [
        "locust",
        "-f", __file__,
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration}s",
        "--headless",
        "--only-summary"
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    # Example: Run quick load test
    run_load_test(
        host="http://localhost:8501",
        users=50,
        spawn_rate=5,
        duration=120
    )
