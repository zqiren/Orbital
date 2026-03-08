// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Service Worker for Orbital PWA Push Notifications

self.addEventListener("push", (event) => {
  if (!event.data) return;

  let data;
  try {
    data = event.data.json();
  } catch {
    data = { title: "Orbital", body: event.data.text() };
  }

  const title = data.title || "Orbital";
  const options = {
    body: data.body || "",
    icon: "/icon-192.png",
    badge: "/icon-192.png",
    tag: data.data?.project_id || "general",
    renotify: true,
    data: data.data || {},
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  const projectId = event.notification.data?.project_id;
  const targetUrl = projectId ? `/?project=${projectId}` : "/";

  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then((windowClients) => {
      for (const client of windowClients) {
        if (client.url.includes(self.location.origin)) {
          client.focus();
          client.postMessage({ type: "navigate", projectId });
          return;
        }
      }
      return clients.openWindow(targetUrl);
    })
  );
});
