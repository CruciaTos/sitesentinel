FROM prom/prometheus:v3.13.3

# Config is baked in (no bind mounts) so the stack can be deployed from a Jenkins container.
COPY prometheus.yml alert-rules.yml /etc/prometheus/
