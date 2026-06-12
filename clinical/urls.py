from django.urls import path

from clinical.views.alerts import SmartAlertsView
from clinical.views.analytics import RiskAssessmentView
from clinical.views.health_story import HealthStoryView
from clinical.views.timeline import TimelineAggregatorView

urlpatterns = [
    path("timeline/", TimelineAggregatorView.as_view(), name="timeline"),
    path("health-story/", HealthStoryView.as_view(), name="health-story"),
    path("alerts/", SmartAlertsView.as_view(), name="alerts"),
    path("analytics/risk-assessment/", RiskAssessmentView.as_view(), name="risk-assessment"),
]
