// Live link from UE5 to the paddy field digital twin API (digital_twin_api.py, /ws/field).
//
// Add the component to the actor that owns the field's HISM, set TargetInstances
// (or bind OnFieldSnapshot yourself), press Play. Per-instance custom data is
// written for every snapshot so the crop material can read it with
// PerInstanceCustomData nodes:
//
//   slot 0  NDVI        [0, 1]
//   slot 1  Stress      [0, 1]
//   slot 2  StressStd   twin uncertainty (std) of stress
//   slot 3  Moisture    [0, 1]  (~0.9 ponded, 0.6 drained but saturated, < 0.6 drying)
//   slot 4  Fungus      [0, 1]  fraction of infected tissue
//   slot 5  Anomaly     1 if the cell was flagged on the current day, else 0
//
// Instance index = row * Cols + col, row 0 = north edge, col 0 = west edge
// (the same convention as /field/status).

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "TwinStreamComponent.generated.h"

class IWebSocket;
class UInstancedStaticMeshComponent;
class FJsonObject;

USTRUCT(BlueprintType)
struct TWINSTREAM_API FTwinFieldSnapshot
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") int32 Seq = 0;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") int32 SimDay = 0;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") FString FieldId;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") FString GrowthStage;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") float Gdd = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") int32 Rows = 0;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") int32 Cols = 0;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") float CellSizeM = 0.f;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") double OriginLat = 0.0;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") double OriginLon = 0.0;

	// Row-major, Rows * Cols entries each.
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<float> Ndvi;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<float> Stress;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<float> StressStd;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<float> Moisture;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<float> Fungus;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<float> WaterMm;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<int32> Zone;

	// Instance indices flagged today, and zones with sprays queued for the next day.
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<int32> Anomalies;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<int32> PendingSprays;
};

USTRUCT(BlueprintType)
struct TWINSTREAM_API FTwinTrackSample
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") float TimeS = 0.f;
	// Continuous grid coordinates: row 0..Rows from the north edge, col 0..Cols from the west edge.
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") float Row = 0.f;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") float Col = 0.f;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") float AltM = 0.f;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") bool bSpraying = false;
};

USTRUCT(BlueprintType)
struct TWINSTREAM_API FTwinDroneTrack
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") FString DroneId;
	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin") TArray<FTwinTrackSample> Samples;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTwinFieldSnapshot, const FTwinFieldSnapshot&, Snapshot);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnTwinSwarmTrack, const FString&, MissionKind, int32, SimDay,
	const TArray<FTwinDroneTrack>&, Tracks);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTwinConnectionChanged, bool, bConnected);

UCLASS(ClassGroup = (DigitalTwin), meta = (BlueprintSpawnableComponent))
class TWINSTREAM_API UTwinStreamComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UTwinStreamComponent();

	/** WebSocket endpoint of digital_twin_api.py. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Digital Twin")
	FString StreamUrl = TEXT("ws://127.0.0.1:8008/ws/field");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Digital Twin")
	bool bAutoConnect = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Digital Twin", meta = (ClampMin = "0.2"))
	float ReconnectDelaySeconds = 2.f;

	/** If set, every snapshot is written into this component's per-instance custom data. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Digital Twin")
	TObjectPtr<UInstancedStaticMeshComponent> TargetInstances;

	UPROPERTY(BlueprintAssignable, Category = "Digital Twin")
	FOnTwinFieldSnapshot OnFieldSnapshot;

	UPROPERTY(BlueprintAssignable, Category = "Digital Twin")
	FOnTwinSwarmTrack OnSwarmTrack;

	UPROPERTY(BlueprintAssignable, Category = "Digital Twin")
	FOnTwinConnectionChanged OnConnectionChanged;

	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin")
	FTwinFieldSnapshot LatestSnapshot;

	UPROPERTY(BlueprintReadOnly, Category = "Digital Twin")
	bool bConnected = false;

	UFUNCTION(BlueprintCallable, Category = "Digital Twin")
	void Connect();

	UFUNCTION(BlueprintCallable, Category = "Digital Twin")
	void Disconnect();

	/** Ask the server for a fresh snapshot now. */
	UFUNCTION(BlueprintCallable, Category = "Digital Twin")
	void RequestSnapshot();

	/** Write a snapshot into per-instance custom data slots 0-5 (see header comment). */
	UFUNCTION(BlueprintCallable, Category = "Digital Twin")
	static void ApplySnapshotToInstances(UInstancedStaticMeshComponent* Instances, const FTwinFieldSnapshot& Snapshot);

	/**
	 * Field-local position in cm from continuous grid coordinates: X = east (col), Y = south (row),
	 * Z = up. Use (Row + 0.5, Col + 0.5) for the centre of instance (Row, Col).
	 */
	UFUNCTION(BlueprintPure, Category = "Digital Twin")
	static FVector GridToLocal(float Row, float Col, float AltM, float CellSizeM);

	/** Interpolate a drone track at TimeS (clamped to the track). */
	UFUNCTION(BlueprintPure, Category = "Digital Twin")
	static FTwinTrackSample SampleTrack(const FTwinDroneTrack& Track, float TimeS);

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
	void HandleMessage(const FString& Text);
	void HandleSnapshot(const TSharedPtr<FJsonObject>& Json);
	void HandleTrack(const TSharedPtr<FJsonObject>& Json);
	void SetConnected(bool bNewConnected);
	void ScheduleReconnect();
	void ReleaseSocket();

	TSharedPtr<IWebSocket> Socket;
	FTimerHandle ReconnectTimer;
	bool bWantConnection = false;
};
