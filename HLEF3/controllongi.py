
class ControlLongi:
    def __init__(self, car_my):
        self.car_my = car_my
        self.car_lead = None

    def set_car_lead(self, lead):
        self.car_lead = lead

    def select_action(self, action_old):
        action = action_old
        if self.car_lead is not None:
            x_diff = self.car_lead.s[0] - self.car_my.s[0]
            v_diff = self.car_lead.s[2] - self.car_my.s[2]
            if x_diff <= 20:
                if v_diff >= 5:
                    action = 1
                    print(self.car_my.id, "slow to cruise")
                else:
                    action = 0
                    print(self.car_my.id, "slow to decel")
        return action
