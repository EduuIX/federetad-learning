import math

class AlertRateHoeffdingTrigger:
    def __init__(self, alpha=0.01, warmup=30):
        self.alpha = alpha
        self.B = warmup
        self.t = 0
        self.sum_alert = 0
        self.count = 0
        self.p0_est = None

    def update(self, alert_bits_count, m_participants):
        self.t += 1
        self.sum_alert += alert_bits_count
        self.count += m_participants
        if self.t == self.B:
            self.p0_est = self.sum_alert / max(1, self.count)
        if self.t <= self.B or self.p0_est is None:
            return False, None, (self.sum_alert / max(1, self.count))
        theta = self.p0_est + math.sqrt(max(0.0, math.log(1/self.alpha)/(2*max(1, m_participants))))
        rhat = alert_bits_count / max(1, m_participants)
        return (rhat >= theta), float(theta), float(self.p0_est)

class AlertRateCUSUMTrigger:
    def __init__(self, warmup=30, p1_guess=0.2, h=5.0, reset_on_trigger=True):
        self.B = warmup
        self.t = 0
        self.sum_alert = 0
        self.count = 0
        self.p0_est = None
        self.p1_guess = p1_guess
        self.h = h
        self.S = 0.0
        self.reset_on_trigger = reset_on_trigger

    def update(self, alert_bits_count, m_participants):
        import math
        self.t += 1
        self.sum_alert += alert_bits_count
        self.count += m_participants
        if self.t == self.B:
            self.p0_est = max(1e-6, min(1-1e-6, self.sum_alert / max(1, self.count)))
        if self.t <= self.B or self.p0_est is None:
            return False, None, (self.sum_alert / max(1, self.count))
        p0 = self.p0_est
        p1 = max(p0 + 1e-3, min(1 - 1e-6, self.p1_guess))
        k = alert_bits_count
        m = max(1, m_participants)
        llr = k*math.log(p1/p0) + (m-k)*math.log((1-p1)/(1-p0))
        self.S = max(0.0, self.S + llr)
        trig = self.S >= self.h
        if trig and self.reset_on_trigger:
            self.S = 0.0
        return trig, float(self.h), float(self.p0_est)
