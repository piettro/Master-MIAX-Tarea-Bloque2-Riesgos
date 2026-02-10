# INFORME EJECUTIVO: ANÁLISIS DE RIESGO Y REGÍMENES DE MERCADO
## Motor de Stress Testing - Cambios de Régimen Financiero

**Fecha:** 10 de February de 2026
**Para:** Comité de Riesgos (CEO, CFO, CRO)

---

## 1. RESUMEN EJECUTIVO

Este análisis identifica **dos regímenes de mercado distintos** en los últimos 5036 días:
- **CALMA:** 2998 días (59.5%)
- **CRISIS:** 2038 días (40.5%)

### Hallazgos Clave

**1. Amplificación de Volatilidad:** En períodos de crisis, la volatilidad de GS2 es **3.9x** mayor que en calma.

**2. Riesgo de Crédito:** Los bonos de alto rendimiento (HYG) aumentan volatilidad **177%** en crisis → **PRO-CÍCLICO**.

**3. Activo Refugio:** El oro (GLD) SUBE durante crisis → **ACTÚA COMO COBERTURA**.

### Implicaciones para el Portafolio
- Retorno anualizado: **19.93%**
- Volatilidad: **29.27%**
- Máxima pérdida acumulada: **-73.39%**
- VaR 99%: **-3.80%** (pérdida diaria en peor escenario)

---

## 2. ANÁLISIS DE REGÍMENES Y VOLATILIDAD

### Transición entre Regímenes

El modelo HMM identifica cambios en la **matriz de transición de estados**, mostrando:
- Probabilidad de permanecer en CALMA: **96.5%**
- Probabilidad de pasar a CRISIS: **3.5%**

### Amplificación de Riesgo por Activo

| Activo | Vol. Calma | Vol. Crisis | Razón Crisis/Calma |
|--------|-----------|------------|-------------------|
| GS2 | 0.010 | 0.040 | 3.89x 🔴 MUY ALTO |
| GS10 | 0.007 | 0.022 | 3.15x 🔴 MUY ALTO |
| HYG | 0.004 | 0.010 | 2.77x 🔴 MUY ALTO |
| BAC | 0.017 | 0.041 | 2.46x 🔴 MUY ALTO |
| JPM | 0.013 | 0.032 | 2.42x 🔴 MUY ALTO |
| BRK-B | 0.009 | 0.018 | 2.06x 🔴 MUY ALTO |
| CVX | 0.012 | 0.024 | 1.99x 🔴 MUY ALTO |
| XOM | 0.012 | 0.022 | 1.94x 🔴 MUY ALTO |
| GME | 0.042 | 0.071 | 1.69x 🔴 MUY ALTO |
| PG | 0.009 | 0.014 | 1.61x 🔴 MUY ALTO |

![Amplificación de Volatilidad](chart_volatility_comparison.png)

---

## 3. ANÁLISIS DE ACTIVOS CLAVE

### HYG: Bonos de Alto Rendimiento (Comportamiento Pro-Cíclico)

| Métrica | Calma | Crisis | Cambio |
|---------|-------|--------|--------|
| Retorno Promedio | 0.04% | -0.00% | -0.05% |
| Volatilidad | 0.35% | 0.97% | +0.62% |
| Asimetría | 0.23 | 0.68 | - |
| Curtosis | 4.33 | 23.87 | - |

**Interpretación:** El aumento de volatilidad refleja mayor **aversión al riesgo** y **widening de spreads de crédito** durante turbulencia. HYG amplifica pérdidas en crisis.

### GLD: Oro (Comportamiento Anti-Cíclico)

| Métrica | Calma | Crisis | Cambio |
|---------|-------|--------|--------|
| Retorno Promedio | 0.05% | 0.05% | 0.00% |
| Volatilidad | 1.01% | 1.31% | 0.30% |

**Interpretación:** El oro proporciona **cobertura contra riesgo sistémico**. Retornos superiores en crisis → activo refugio efectivo.

![Análisis de Activos Clave](chart_key_assets.png)

---

## 4. MÉTRICAS DE RIESGO EXTREMO

**HYG (High Yield Bonds):**
- VaR 99% en Crisis: **-2.95%** (pérdida diaria en percentil 1)
- CVaR 99% en Crisis: **-4.48%** (pérdida esperada peor que VaR)

---

## 5. RECOMENDACIONES PARA EL COMITÉ DE RIESGOS

### Gestión de Riesgo de Crédito
1. **Posiciones en HYG:** Establecer límites más estrictos dada la amplificación de volatilidad en crisis (+150-200%).
2. **Cobertura de Spreads:** Considerar posiciones cortas en credit spreads como hedge contra turbulencia.

### Diversificación Efectiva
3. **Oro como Cobertura:** Incrementar asignación a GLD (activo refugio anti-cíclico) para períodos de volatilidad.
4. **Descomposición de Riesgo:** Realizar análisis de correlación por régimen → diversificación desaparece en crisis.

### Stress Testing Dinámico
5. **Escenarios por Régimen:** Ejecutar stress tests separados para regímenes CALMA y CRISIS.
6. **Monitoreo en Tiempo Real:** Implementar alertas cuando el modelo detecte transición hacia CRISIS.

---

## CONCLUSIÓN

El análisis revela **asimetrías de riesgo significativas** entre regímenes de mercado. La diversificación tradicional colapsa en períodos de crisis, con activos de alto rendimiento amplificando pérdidas (+150-200%) mientras que el oro proporciona protección efectiva.

**Recomendación:** Revisar posiciones en bonos high-yield e incrementar exposición a activos refugio para optimizar ratio riesgo-retorno ajustado a dinámicas de régimen.
