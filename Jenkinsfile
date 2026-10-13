// SiteSentinel local CI/CD pipeline (runs in jenkins/docker-compose.yml, uses the host's Docker daemon).
// Lint & test -> train & validate the model -> build -> push to localhost:5000 -> deploy with
// docker compose -> smoke test -> automatic rollback to the previously running image on failure.
pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 30, unit: 'MINUTES')
    }

    parameters {
        booleanParam(name: 'SIMULATE_FAILED_DEPLOY', defaultValue: false,
            description: 'Make the smoke test fail after deploying, to demonstrate the automatic rollback.')
    }

    environment {
        REGISTRY = 'localhost:5000'
        IMAGE = 'localhost:5000/sitesentinel-backend'
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.TAG = 'sha-' + sh(script: 'git rev-parse --short=7 HEAD', returnStdout: true).trim()
                    currentBuild.displayName = "#${env.BUILD_NUMBER} ${env.TAG}"
                }
                echo "Building ${env.IMAGE}:${env.TAG}"
            }
        }

        stage('Lint & Unit Tests') {
            steps {
                sh '''
                    rm -rf reports
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install -q -r backend/requirements-dev.txt -r inference/requirements-test.txt
                    (cd backend && flake8 . && pytest --junitxml=../reports/backend-junit.xml)
                    (cd inference && flake8 . && pytest --junitxml=../reports/inference-junit.xml)
                '''
            }
        }

        stage('Train & Validate Model') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pip install -q -r ml/requirements.txt pytest flake8
                    (cd ml && flake8 . && pytest --junitxml=../reports/ml-junit.xml)
                    mkdir -p reports
                    python ml/train.py --out backend/model/ppe_model.json --report ml/reports/metrics.json \
                        --summary reports/model-summary.md
                '''
            }
        }

        stage('Build Image') {
            steps {
                sh '''
                    docker build --target test -t sitesentinel-backend:test backend
                    docker build --build-arg APP_VERSION="$TAG" -t "$IMAGE:$TAG" backend
                '''
            }
        }

        stage('Push to Registry') {
            steps {
                // Only the commit tag here; "latest" is moved once the release has passed its smoke test.
                sh '''
                    docker push "$IMAGE:$TAG"
                    curl -fsS http://registry:5000/v2/sitesentinel-backend/tags/list || true
                '''
            }
        }

        stage('Deploy') {
            steps {
                script {
                    env.PREVIOUS_IMAGE = sh(returnStdout: true, script: '''
                        c="$(docker compose ps -q backend 2>/dev/null || true)"
                        if [ -n "$c" ]; then docker inspect --format '{{.Config.Image}}' "$c"; fi
                    ''').trim()
                    env.DEPLOY_STARTED = 'true'
                    echo "Previously running image: ${env.PREVIOUS_IMAGE ?: 'none'}"
                }
                sh 'bash scripts/deploy.sh "$IMAGE" "$TAG"'
            }
        }

        stage('Smoke Test') {
            steps {
                script {
                    env.EXPECTED_VERSION = params.SIMULATE_FAILED_DEPLOY ? 'simulated-failure' : env.TAG
                }
                sh 'bash scripts/smoke-test.sh http://host.docker.internal:8080 "$EXPECTED_VERSION"'
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/*.xml'
            archiveArtifacts allowEmptyArchive: true,
                artifacts: 'backend/model/ppe_model.json, ml/reports/metrics.json, reports/**'
        }
        failure {
            script {
                if (env.DEPLOY_STARTED == 'true') {
                    if (env.PREVIOUS_IMAGE && env.PREVIOUS_IMAGE != env.IMAGE + ':' + env.TAG) {
                        sh 'bash scripts/rollback.sh "$PREVIOUS_IMAGE"'
                        echo "ROLLED BACK to ${env.PREVIOUS_IMAGE}"
                    } else {
                        echo 'Deploy failed and there is no previous image to roll back to.'
                    }
                }
            }
        }
        success {
            sh '''
                docker tag "$IMAGE:$TAG" "$IMAGE:latest"
                docker push "$IMAGE:latest"
            '''
            echo "DEPLOYED ${env.IMAGE}:${env.TAG}: app http://localhost:8080, Grafana http://localhost:3000, Prometheus http://localhost:9090"
        }
    }
}
