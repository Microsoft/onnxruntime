<?xml version="1.0" encoding="utf-8"?>
<!-- TODO: 
1. Find and replace "OrtTraceLoggingProvider" with your component name.
2. See TODO below to update GUID for your event provider
-->
<WindowsPerformanceRecorder Version="1.0" Author="Microsoft Corporation"
    Copyright="Microsoft Corporation" Company="Microsoft Corporation">
  <Profiles>
    <EventCollector Id="EventCollector_OrtTraceLoggingProvider"
      Name="OrtTraceLoggingProviderCollector">
      <BufferSize Value="65536" />
      <Buffers Value="10" PercentageOfTotalMemory="true"/>
    </EventCollector>

    <EventProvider Id="EventProvider_OrtTraceLoggingProvider"
      Name="3a26b1ff-7484-7484-7484-15261f42614d" />
    <Profile Id="OrtTraceLoggingProvider.Verbose.File"
      Name="OrtTraceLoggingProvider" Description="OrtTraceLoggingProvider"
      LoggingMode="File" DetailLevel="Verbose">
      <Collectors>
        <EventCollectorId Value="EventCollector_OrtTraceLoggingProvider">
          <EventProviders>
            <EventProviderId Value="EventProvider_OrtTraceLoggingProvider" />
          </EventProviders>
        </EventCollectorId>
      </Collectors>
    </Profile>

    <Profile Id="OrtTraceLoggingProvider.Light.File"
      Name="OrtTraceLoggingProvider"
      Description="OrtTraceLoggingProvider"
      Base="OrtTraceLoggingProvider.Verbose.File"
      LoggingMode="File"
      DetailLevel="Light" />

    <Profile Id="OrtTraceLoggingProvider.Verbose.Memory"
      Name="OrtTraceLoggingProvider"
      Description="OrtTraceLoggingProvider"
      Base="OrtTraceLoggingProvider.Verbose.File"
      LoggingMode="Memory"
      DetailLevel="Verbose" />

    <Profile Id="OrtTraceLoggingProvider.Light.Memory"
      Name="OrtTraceLoggingProvider"
      Description="OrtTraceLoggingProvider"
      Base="OrtTraceLoggingProvider.Verbose.File"
      LoggingMode="Memory"
      DetailLevel="Light" />

  </Profiles>
</WindowsPerformanceRecorder>
