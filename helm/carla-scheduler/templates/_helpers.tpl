{{- define "carla-scheduler.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "carla-scheduler.fullname" -}}
{{- printf "%s" (include "carla-scheduler.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "carla-scheduler.labels" -}}
app.kubernetes.io/name: {{ include "carla-scheduler.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "carla-scheduler.selectorLabels" -}}
app.kubernetes.io/name: {{ include "carla-scheduler.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "carla-scheduler.serviceAccountName" -}}
{{- default (include "carla-scheduler.fullname" .) .Values.serviceAccount.name -}}
{{- end -}}
