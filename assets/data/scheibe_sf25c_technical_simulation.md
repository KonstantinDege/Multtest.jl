# Technical Specifications & High-Level Simulation: Scheibe SF 25 C Falke

The **Scheibe SF 25 C Falke** is a legendary two-seat touring motor glider (Reisemotorsegler) of mixed construction. Below is a comprehensive breakdown of its technical parameters followed by a high-level aerodynamic and cruise flight simulation model.

---

## 1. Technical Data & Performance Metrics

### Dimensions & Weights
* **Wingspan:** 15.30 m
* **Length:** 7.60 m
* **Wing Area ($S$):** 18.20 m²
* **Aspect Ratio ($AR$):** 13.80
* **Empty Weight:** ~415 – 450 kg (depending on configuration/equipment)
* **Maximum Takeoff Weight (MTOW):** 
  * *Standard variants:* 650 kg
  * *Modern Rotax upgrades:* Up to 690 kg / 800 kg

### Flight Performance
* **Never Exceed Speed ($V_{NE}$):** 190 km/h
* **Cruise Speed ($V_C$):** ~135 – 150 km/h (engine dependent)
* **Stall Speed ($V_S$):** 60 km/h
* **Best Glide Ratio ($L/D_{max}$):** 23 – 24 (at ~90 km/h)
* **Minimum Sink Rate:** 1.1 m/s (at ~75 km/h)
* **Service Ceiling:** 4,300 m

### Propulsion & Fuel
* **Common Engine Configurations:**
  * **Limbach SL 1700 / L 2000:** 64 to 80 hp
  * **Sauer S 2100:** 80 hp
  * **Rotax 912 A/F/S:** 80 to 100 hp (highly popular modern retrofit)
* **Fuel Capacity:** Standard tank is 55 liters

---

## 2. High-Level Flight Mechanics Simulation

To conceptualize the flight characteristics of the SF 25 C in both soaring (engine off) and cruise (engine on) configurations, we can define a simplified mathematical model.

### Simulation Matrix: Engine-Off Gliding Performance

| Parameter / Velocity | 75 km/h (Min Sink) | 90 km/h (Best Glide) | 120 km/h (Glider Cruise) |
| :--- | :--- | :--- | :--- |
| **Lift Coefficient ($C_L$)** | ~0.95 | ~0.66 | ~0.37 |
| **Drag Coefficient ($C_D$)** | ~0.043 | ~0.028 | ~0.021 |
| **Glide Ratio ($L/D$)** | ~22.1 | ~23.6 | ~17.6 |
| **Sink Rate ($v_z$)** | 1.10 m/s | 1.15 m/s | 1.89 m/s |

### Simulation Matrix: Powered Cruise (Standard Atmosphere, Sea Level)

For a standard **Limbach 80 hp** variant operating at 75% MCP (Maximum Continuous Power = ~60 hp / 44.7 kW) with a propeller efficiency ($\eta$) of 0.70:

* **Available Thrust Power ($P_A$):** $44.7 \text{ kW} \times 0.70 = 31.3 \text{ kW}$
* **Maximum Theoretical Cruise Velocity ($V_{max\_cruise}$):** ~145 km/h
* **Fuel Burn Profile (Limbach L 2000):** ~12–14 Liters/hour at economy cruise
* **Theoretical Safe Range (55L Tank, 45-min reserve):** ~450 – 520 km

---

## 3. High-Level Summary For Pilots
The SF 25 C exhibits forgiving, benign handling characteristics. Due to its relatively low wing loading (~35.7 kg/m² at MTOW), it climbs effectively in thermal air despite its modest glide ratio compared to modern composite sailplanes. Under power, it serves as a highly economical cross-country trainer.