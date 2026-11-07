#!/bin/bash
# Runs as root only long enough to restrict egress inside this container's own
# network namespace, then drops to an unprivileged user with no capabilities.
# Any failure stops the container: it must never start with egress unrestricted.
set -euo pipefail

blocked=${ORBITAL_BLOCKED_NETS-10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10 169.254.0.0/16}

/usr/sbin/iptables -F OUTPUT
/usr/sbin/iptables -A OUTPUT -o lo -j ACCEPT
/usr/sbin/iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
for net in $blocked; do
  /usr/sbin/iptables -A OUTPUT -d "$net" -j REJECT --reject-with icmp-admin-prohibited
done

exec setpriv --reuid=10001 --regid=10001 --clear-groups \
  --inh-caps=-all --ambient-caps=-all --bounding-set=-all --no-new-privs \
  /usr/bin/tini -- /bin/bash -c \
  'mkdir -p "$HOME/.orbital/logs" && exec /usr/bin/supervisord -n -c /etc/orbital/supervisord.conf'
