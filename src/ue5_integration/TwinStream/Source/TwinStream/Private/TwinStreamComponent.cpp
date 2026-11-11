#include "TwinStreamComponent.h"

#include "Algo/BinarySearch.h"
#include "Async/Async.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "IWebSocket.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "TimerManager.h"
#include "WebSocketsModule.h"

DEFINE_LOG_CATEGORY_STATIC(LogTwinStream, Log, All);

namespace TwinStream
{
	constexpr int32 NumCustomDataFloats = 6;

	static void ReadFloats(const TSharedPtr<FJsonObject>& Json, const TCHAR* Key, TArray<float>& Out)
	{
		Out.Reset();
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (Json->TryGetArrayField(Key, Values) && Values)
		{
			Out.Reserve(Values->Num());
			for (const TSharedPtr<FJsonValue>& V : *Values)
			{
				Out.Add(V.IsValid() ? static_cast<float>(V->AsNumber()) : 0.f);
			}
		}
	}

	static void ReadInts(const TSharedPtr<FJsonObject>& Json, const TCHAR* Key, TArray<int32>& Out)
	{
		Out.Reset();
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (Json->TryGetArrayField(Key, Values) && Values)
		{
			Out.Reserve(Values->Num());
			for (const TSharedPtr<FJsonValue>& V : *Values)
			{
				Out.Add(V.IsValid() ? static_cast<int32>(V->AsNumber()) : 0);
			}
		}
	}

	static double ReadNumber(const TSharedPtr<FJsonObject>& Json, const TCHAR* Key, double Default = 0.0)
	{
		double Value = Default;
		Json->TryGetNumberField(Key, Value);
		return Value;
	}
}

UTwinStreamComponent::UTwinStreamComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UTwinStreamComponent::BeginPlay()
{
	Super::BeginPlay();
	if (bAutoConnect)
	{
		Connect();
	}
}

void UTwinStreamComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Disconnect();
	Super::EndPlay(EndPlayReason);
}

void UTwinStreamComponent::Connect()
{
	bWantConnection = true;
	if (Socket.IsValid() && Socket->IsConnected())
	{
		return;
	}
	ReleaseSocket();

	FWebSocketsModule& Module = FModuleManager::LoadModuleChecked<FWebSocketsModule>(TEXT("WebSockets"));
	Socket = Module.CreateWebSocket(StreamUrl);

	// Socket callbacks may arrive off the game thread; hop back before touching UObjects.
	TWeakObjectPtr<UTwinStreamComponent> WeakThis(this);

	Socket->OnConnected().AddLambda([WeakThis]()
	{
		AsyncTask(ENamedThreads::GameThread, [WeakThis]()
		{
			if (WeakThis.IsValid())
			{
				UE_LOG(LogTwinStream, Log, TEXT("Connected to %s"), *WeakThis->StreamUrl);
				WeakThis->SetConnected(true);
			}
		});
	});

	Socket->OnConnectionError().AddLambda([WeakThis](const FString& Error)
	{
		AsyncTask(ENamedThreads::GameThread, [WeakThis, Error]()
		{
			if (WeakThis.IsValid())
			{
				UE_LOG(LogTwinStream, Warning, TEXT("Connection error: %s"), *Error);
				WeakThis->SetConnected(false);
				WeakThis->ScheduleReconnect();
			}
		});
	});

	Socket->OnClosed().AddLambda([WeakThis](int32 StatusCode, const FString& Reason, bool bWasClean)
	{
		AsyncTask(ENamedThreads::GameThread, [WeakThis, StatusCode, Reason]()
		{
			if (WeakThis.IsValid())
			{
				UE_LOG(LogTwinStream, Log, TEXT("Closed (%d): %s"), StatusCode, *Reason);
				WeakThis->SetConnected(false);
				WeakThis->ScheduleReconnect();
			}
		});
	});

	Socket->OnMessage().AddLambda([WeakThis](const FString& Message)
	{
		AsyncTask(ENamedThreads::GameThread, [WeakThis, Message]()
		{
			if (WeakThis.IsValid())
			{
				WeakThis->HandleMessage(Message);
			}
		});
	});

	Socket->Connect();
}

void UTwinStreamComponent::Disconnect()
{
	bWantConnection = false;
	if (UWorld* World = GetWorld())
	{
		World->GetTimerManager().ClearTimer(ReconnectTimer);
	}
	ReleaseSocket();
	SetConnected(false);
}

void UTwinStreamComponent::ReleaseSocket()
{
	if (Socket.IsValid())
	{
		// Detach first so a stale socket can never trigger another reconnect.
		Socket->OnConnected().Clear();
		Socket->OnConnectionError().Clear();
		Socket->OnClosed().Clear();
		Socket->OnMessage().Clear();
		if (Socket->IsConnected())
		{
			Socket->Close();
		}
		Socket.Reset();
	}
}

void UTwinStreamComponent::RequestSnapshot()
{
	if (Socket.IsValid() && Socket->IsConnected())
	{
		Socket->Send(TEXT("{\"type\":\"snapshot\"}"));
	}
}

void UTwinStreamComponent::SetConnected(bool bNewConnected)
{
	if (bConnected != bNewConnected)
	{
		bConnected = bNewConnected;
		OnConnectionChanged.Broadcast(bConnected);
	}
}

void UTwinStreamComponent::ScheduleReconnect()
{
	UWorld* World = GetWorld();
	if (!bWantConnection || !World || World->GetTimerManager().IsTimerActive(ReconnectTimer))
	{
		return;
	}
	World->GetTimerManager().SetTimer(ReconnectTimer, this, &UTwinStreamComponent::Connect,
		FMath::Max(0.2f, ReconnectDelaySeconds), false);
}

void UTwinStreamComponent::HandleMessage(const FString& Text)
{
	TSharedPtr<FJsonObject> Json;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
	if (!FJsonSerializer::Deserialize(Reader, Json) || !Json.IsValid())
	{
		UE_LOG(LogTwinStream, Warning, TEXT("Ignoring malformed message (%d chars)"), Text.Len());
		return;
	}

	FString Type;
	Json->TryGetStringField(TEXT("type"), Type);
	if (Type == TEXT("field.snapshot"))
	{
		HandleSnapshot(Json);
	}
	else if (Type == TEXT("swarm.track"))
	{
		HandleTrack(Json);
	}
	else if (Type == TEXT("swarm.positions"))
	{
		HandlePositions(Json);
	}
	else if (Type == TEXT("twin.status"))
	{
		HandleStatus(Json);
	}
}

void UTwinStreamComponent::HandleSnapshot(const TSharedPtr<FJsonObject>& Json)
{
	using namespace TwinStream;
	FTwinFieldSnapshot S;
	S.Seq = static_cast<int32>(ReadNumber(Json, TEXT("seq")));
	S.SimDay = static_cast<int32>(ReadNumber(Json, TEXT("sim_day")));
	S.Gdd = static_cast<float>(ReadNumber(Json, TEXT("gdd")));
	Json->TryGetStringField(TEXT("field_id"), S.FieldId);
	Json->TryGetStringField(TEXT("growth_stage"), S.GrowthStage);

	const TSharedPtr<FJsonObject>* Grid = nullptr;
	if (Json->TryGetObjectField(TEXT("grid"), Grid) && Grid && Grid->IsValid())
	{
		S.Rows = static_cast<int32>(ReadNumber(*Grid, TEXT("rows")));
		S.Cols = static_cast<int32>(ReadNumber(*Grid, TEXT("cols")));
		S.CellSizeM = static_cast<float>(ReadNumber(*Grid, TEXT("cell_size_m")));
		S.OriginLat = ReadNumber(*Grid, TEXT("origin_lat"));
		S.OriginLon = ReadNumber(*Grid, TEXT("origin_lon"));
	}

	ReadFloats(Json, TEXT("ndvi"), S.Ndvi);
	ReadFloats(Json, TEXT("stress"), S.Stress);
	ReadFloats(Json, TEXT("stress_std"), S.StressStd);
	ReadFloats(Json, TEXT("moisture"), S.Moisture);
	ReadFloats(Json, TEXT("fungus"), S.Fungus);
	ReadFloats(Json, TEXT("water_mm"), S.WaterMm);
	ReadInts(Json, TEXT("zone"), S.Zone);
	ReadInts(Json, TEXT("anomalies"), S.Anomalies);
	ReadInts(Json, TEXT("pending_sprays"), S.PendingSprays);
	ReadFloats(Json, TEXT("truth_ndvi"), S.TruthNdvi);
	ReadFloats(Json, TEXT("truth_stress"), S.TruthStress);
	ReadFloats(Json, TEXT("truth_fungus"), S.TruthFungus);

	LatestSnapshot = MoveTemp(S);
	if (TargetInstances)
	{
		ApplySnapshotToInstances(TargetInstances, LatestSnapshot);
	}
	if (TruthInstances && LatestSnapshot.TruthStress.Num() > 0)
	{
		ApplyTruthToInstances(TruthInstances, LatestSnapshot);
	}
	OnFieldSnapshot.Broadcast(LatestSnapshot);
}

void UTwinStreamComponent::HandleTrack(const TSharedPtr<FJsonObject>& Json)
{
	FString Kind;
	Json->TryGetStringField(TEXT("kind"), Kind);
	const int32 SimDay = static_cast<int32>(TwinStream::ReadNumber(Json, TEXT("sim_day")));

	TArray<FTwinDroneTrack> Tracks;
	const TSharedPtr<FJsonObject>* Drones = nullptr;
	if (Json->TryGetObjectField(TEXT("drones"), Drones) && Drones && Drones->IsValid())
	{
		for (const auto& Pair : (*Drones)->Values)
		{
			FTwinDroneTrack Track;
			Track.DroneId = Pair.Key;
			if (Pair.Value.IsValid())
			{
				// Each sample: [t_s, row, col, alt_m, spraying]
				for (const TSharedPtr<FJsonValue>& SampleValue : Pair.Value->AsArray())
				{
					const TArray<TSharedPtr<FJsonValue>>& F = SampleValue->AsArray();
					if (F.Num() < 5)
					{
						continue;
					}
					FTwinTrackSample Sample;
					Sample.TimeS = static_cast<float>(F[0]->AsNumber());
					Sample.Row = static_cast<float>(F[1]->AsNumber());
					Sample.Col = static_cast<float>(F[2]->AsNumber());
					Sample.AltM = static_cast<float>(F[3]->AsNumber());
					Sample.bSpraying = F[4]->AsNumber() > 0.5;
					Track.Samples.Add(Sample);
				}
			}
			Tracks.Add(MoveTemp(Track));
		}
	}
	OnSwarmTrack.Broadcast(Kind, SimDay, Tracks);
}

void UTwinStreamComponent::HandlePositions(const TSharedPtr<FJsonObject>& Json)
{
	const float TimeS = static_cast<float>(TwinStream::ReadNumber(Json, TEXT("t_s")));
	TArray<FTwinDronePosition> Drones;
	const TSharedPtr<FJsonObject>* Obj = nullptr;
	if (Json->TryGetObjectField(TEXT("drones"), Obj) && Obj && Obj->IsValid())
	{
		for (const auto& Pair : (*Obj)->Values)
		{
			// [row, col, alt_m, spraying, heading_deg | null]
			if (!Pair.Value.IsValid())
			{
				continue;
			}
			const TArray<TSharedPtr<FJsonValue>>& F = Pair.Value->AsArray();
			if (F.Num() < 5)
			{
				continue;
			}
			FTwinDronePosition P;
			P.DroneId = Pair.Key;
			P.Row = static_cast<float>(F[0]->AsNumber());
			P.Col = static_cast<float>(F[1]->AsNumber());
			P.AltM = static_cast<float>(F[2]->AsNumber());
			P.bSpraying = F[3]->AsNumber() > 0.5;
			P.bHasHeading = F[4].IsValid() && F[4]->Type == EJson::Number;
			P.HeadingDeg = P.bHasHeading ? static_cast<float>(F[4]->AsNumber()) : 0.f;
			Drones.Add(P);
		}
	}
	OnDronePositions.Broadcast(TimeS, Drones);
}

void UTwinStreamComponent::HandleStatus(const TSharedPtr<FJsonObject>& Json)
{
	FString Phase, Text;
	Json->TryGetStringField(TEXT("phase"), Phase);
	Json->TryGetStringField(TEXT("text"), Text);
	LatestStatusText = Text;
	OnStatus.Broadcast(Phase, Text, static_cast<int32>(TwinStream::ReadNumber(Json, TEXT("sim_day"))));
}

void UTwinStreamComponent::ApplyTruthToInstances(UInstancedStaticMeshComponent* Instances,
	const FTwinFieldSnapshot& Snapshot)
{
	FTwinFieldSnapshot Truth = Snapshot;
	Truth.Ndvi = Snapshot.TruthNdvi;
	Truth.Stress = Snapshot.TruthStress;
	Truth.Fungus = Snapshot.TruthFungus;
	Truth.StressStd.Init(0.f, Snapshot.TruthStress.Num());   // reality has no uncertainty
	Truth.Anomalies.Reset();
	ApplySnapshotToInstances(Instances, Truth);
}

void UTwinStreamComponent::ApplySnapshotToInstances(UInstancedStaticMeshComponent* Instances,
	const FTwinFieldSnapshot& Snapshot)
{
	if (!Instances)
	{
		return;
	}
	if (Instances->NumCustomDataFloats < TwinStream::NumCustomDataFloats)
	{
		Instances->SetNumCustomDataFloats(TwinStream::NumCustomDataFloats);
	}

	TSet<int32> Anomalous(Snapshot.Anomalies);
	const int32 Count = FMath::Min(Instances->GetInstanceCount(), Snapshot.Ndvi.Num());
	auto At = [](const TArray<float>& A, int32 I) { return A.IsValidIndex(I) ? A[I] : 0.f; };

	for (int32 I = 0; I < Count; ++I)
	{
		Instances->SetCustomDataValue(I, 0, At(Snapshot.Ndvi, I), false);
		Instances->SetCustomDataValue(I, 1, At(Snapshot.Stress, I), false);
		Instances->SetCustomDataValue(I, 2, At(Snapshot.StressStd, I), false);
		Instances->SetCustomDataValue(I, 3, At(Snapshot.Moisture, I), false);
		Instances->SetCustomDataValue(I, 4, At(Snapshot.Fungus, I), false);
		Instances->SetCustomDataValue(I, 5, Anomalous.Contains(I) ? 1.f : 0.f, false);
	}
	Instances->MarkRenderStateDirty();
}

FVector UTwinStreamComponent::GridToLocal(float Row, float Col, float AltM, float CellSizeM)
{
	return FVector(Col * CellSizeM * 100.f, Row * CellSizeM * 100.f, AltM * 100.f);
}

FTwinTrackSample UTwinStreamComponent::SampleTrack(const FTwinDroneTrack& Track, float TimeS)
{
	const TArray<FTwinTrackSample>& S = Track.Samples;
	if (S.Num() == 0)
	{
		return FTwinTrackSample();
	}
	if (TimeS <= S[0].TimeS)
	{
		return S[0];
	}
	if (TimeS >= S.Last().TimeS)
	{
		return S.Last();
	}
	// Samples are time-ordered; find the bracketing pair.
	int32 Hi = Algo::LowerBoundBy(S, TimeS, &FTwinTrackSample::TimeS);
	Hi = FMath::Clamp(Hi, 1, S.Num() - 1);
	const FTwinTrackSample& A = S[Hi - 1];
	const FTwinTrackSample& B = S[Hi];
	const float Span = FMath::Max(B.TimeS - A.TimeS, KINDA_SMALL_NUMBER);
	const float Alpha = FMath::Clamp((TimeS - A.TimeS) / Span, 0.f, 1.f);

	FTwinTrackSample Out;
	Out.TimeS = TimeS;
	Out.Row = FMath::Lerp(A.Row, B.Row, Alpha);
	Out.Col = FMath::Lerp(A.Col, B.Col, Alpha);
	Out.AltM = FMath::Lerp(A.AltM, B.AltM, Alpha);
	Out.bSpraying = Alpha < 0.5f ? A.bSpraying : B.bSpraying;
	return Out;
}
