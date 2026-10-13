# Local Jenkins for the SiteSentinel pipeline: Jenkins LTS + Docker CLI/Compose + Python,
# configured as code (admin user and the "sitesentinel" pipeline job).
FROM jenkins/jenkins:2.568.3-lts-jdk21

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*
# Docker CLI with the compose and buildx plugins; builds and deploys use the host's Docker daemon.
COPY --from=docker:29-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker:29-cli /usr/local/libexec/docker/cli-plugins /usr/local/libexec/docker/cli-plugins
# The repo is bind-mounted from the host, so its owner differs from the Jenkins user.
RUN git config --system --add safe.directory '*'

USER jenkins
COPY --chown=jenkins:jenkins plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
COPY --chown=jenkins:jenkins casc.yaml /var/jenkins_conf/casc.yaml

ENV CASC_JENKINS_CONFIG=/var/jenkins_conf/casc.yaml
ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false -Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true"
