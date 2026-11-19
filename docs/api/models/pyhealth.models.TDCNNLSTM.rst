pyhealth.models.TDCNNLSTM
===================================

CNN-LSTM value network over irregular measurement tuples, trained with
temporal-difference (TD) learning against a soft-updated target network, or
with supervised targets as a baseline. Reproduces Frost, Li & Harris,
`Robust Real-Time Mortality Prediction in the Intensive Care Unit using
Temporal Difference Learning <https://proceedings.mlr.press/v259/frost25a.html>`_
(ML4H 2024). Use it with :class:`pyhealth.tasks.RealTimeICUMortalityMIMIC4`
and the stock :class:`pyhealth.trainer.Trainer`; see
``examples/mimic4_realtime_icu_mortality_td_cnn_lstm.py``.

.. autoclass:: pyhealth.models.TDCNNLSTMEncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TDCNNLSTM
    :members:
    :undoc-members:
    :show-inheritance:
