"""
Đường truyền giữa tòa nhà và twin.

- MqttBus: MQTT thật (chuẩn IoT). Đặt trong .env:
      MQTT_HOST=broker.hivemq.com   (hoặc broker riêng / HiveMQ Cloud)
      MQTT_PORT=1883  [MQTT_USER, MQTT_PASS, MQTT_TLS=1]  MQTT_PREFIX=elevtwin/<ma_nhom>
- LocalBus: chạy trong cùng tiến trình, không cần mạng (mặc định khi chưa đặt MQTT_HOST).

Hai bên chỉ trao đổi tin nhắn JSON theo chủ đề; twin không đọc được bên trong tòa nhà.
Chủ đề (sau tiền tố):
  state      tòa nhà -> twin   trạng thái cabin + nút gọi tầng, mỗi giây
  stop       tòa nhà -> twin   mỗi lần mở cửa: lượt lên/xuống (hồng ngoại), tải (cân), còn người chờ
  truth      tòa nhà -> web    số liệu thật của tòa nhà ảo (twin KHÔNG đọc) để đối chiếu
  twin       twin -> web       trạng thái twin: chế độ AI, đề xuất, dự báo
  cmd/roles  twin -> tòa nhà   lệnh phân vai cabin
  cmd/fault  web -> tòa nhà    giả lập sự cố cabin
  cmd/control web -> tòa nhà   'closed' (áp dụng lệnh twin) | 'advisory' (chỉ đề xuất)
  report     người dùng -> twin  báo cáo hàng chờ từ form QR
"""
from __future__ import annotations

import json
import os
import threading
import uuid


def prefix():
    return os.environ.get("MQTT_PREFIX", "elevtwin/demo").rstrip("/")


class LocalBus:
    kind = "local"

    def __init__(self):
        self.subs = []
        self.lock = threading.Lock()

    def subscribe(self, topic, callback):
        with self.lock:
            self.subs.append((topic, callback))

    def publish(self, topic, payload):
        data = json.loads(json.dumps(payload))      # sao chép như khi đi qua mạng
        with self.lock:
            subs = list(self.subs)
        for t, cb in subs:
            if t == topic or (t.endswith("#") and topic.startswith(t[:-1])):
                cb(topic, data)

    def close(self):
        self.subs.clear()


class MqttBus:
    kind = "mqtt"

    def __init__(self, host=None, port=None):
        import paho.mqtt.client as mqtt
        self.host = host or os.environ["MQTT_HOST"]
        self.port = int(port or os.environ.get("MQTT_PORT", 1883))
        self.subs = []
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"twin-{uuid.uuid4().hex[:8]}")
        if os.environ.get("MQTT_USER"):
            self.client.username_pw_set(os.environ["MQTT_USER"], os.environ.get("MQTT_PASS", ""))
        if os.environ.get("MQTT_TLS") == "1":
            self.client.tls_set()
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, props=None):
        for t, _ in self.subs:
            client.subscribe(t, qos=0)

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
        except ValueError:
            return
        for t, cb in list(self.subs):
            if t == msg.topic or (t.endswith("#") and msg.topic.startswith(t[:-1])):
                cb(msg.topic, data)

    def subscribe(self, topic, callback):
        self.subs.append((topic, callback))
        self.client.subscribe(topic, qos=0)

    def publish(self, topic, payload):
        self.client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=0)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


def get_bus():
    if os.environ.get("MQTT_HOST"):
        return MqttBus()
    return LocalBus()
