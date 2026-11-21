class DroneContactMonitor:
    def __init__(
        self,
        *,
        body_path,
        sensor_name="contact_sensor",
        min_threshold=0.0,
        max_threshold=10000000.0,
        radius=-1.0,
    ):
        from isaacsim.sensors.physics import ContactSensor

        self.sensor_path = f"{body_path}/{sensor_name}"
        self.sensor = ContactSensor(
            prim_path=self.sensor_path,
            name="drone_contact_sensor",
            min_threshold=float(min_threshold),
            max_threshold=float(max_threshold),
            radius=float(radius),
        )
        self.sensor.add_raw_contact_data_to_frame()
        self.was_in_contact = False
        self.max_force = 0.0
        self.contact_count = 0

    def initialize(self):
        self.sensor.initialize()

    def update(self, *, phase, step):
        frame = self.sensor.get_current_frame()
        in_contact = bool(frame.get("in_contact", False))
        force = float(frame.get("force", 0.0))
        num_contacts = int(frame.get("number_of_contacts", 0))
        self.max_force = max(self.max_force, force)

        if in_contact and not self.was_in_contact:
            self.contact_count += 1

        contacts = []
        for contact in frame.get("contacts", []):
            contacts.append(
                {
                    "body0": contact.get("body0"),
                    "body1": contact.get("body1"),
                }
            )

        new_contact = in_contact and not self.was_in_contact
        self.was_in_contact = in_contact
        return {
            "in_contact": in_contact,
            "new_contact": new_contact,
            "force": force,
            "number_of_contacts": num_contacts,
            "contacts": contacts,
            "max_force": self.max_force,
            "contact_count": self.contact_count,
        }
