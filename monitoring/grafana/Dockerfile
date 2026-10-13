FROM grafana/grafana:13.2.2

# Demo defaults: admin/admin, read-only anonymous access, the SiteSentinel dashboard as home page.
ENV GF_SECURITY_ADMIN_USER=admin \
    GF_SECURITY_ADMIN_PASSWORD=admin \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/sitesentinel.json \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false

# Provisioning and dashboards are baked in (no bind mounts).
COPY provisioning /etc/grafana/provisioning
COPY dashboards /var/lib/grafana/dashboards
