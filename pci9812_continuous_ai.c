// PCI_9812 : Continuous AI  (syntax-fixed)
#include <stdio.h>
#include "dask.h"

// ── Constants ────────────────────────────────────────────────────────────────
#define CardNumber   0
#define LastADChan   3               // CH0–CH3 (4 channels)
#define ADChanCount  4
#define ScanCount    65536           // scans per half-buffer

// At 20 MS/s: half = 65536 / 20 000 000 ≈ 3.3 ms
// At 20 kS/s: half = 65536 / 20 000    ≈ 3.3 s  (lower rate needs smaller ScanCount)
#define SampleRate   20000000.0      // S/s — change to 20000.0 for 20 kS/s

#define ClkSel       (P9812_AD2_GT_PCI | P9812_CLKSRC_INT)
#define TriggerLevel 127             // 0x7F
#define PostCount    0

// ── Variables ────────────────────────────────────────────────────────────────
I16     cardID    = -1;
I16     err       = 0;
BOOLEAN HalfReady = 0;
BOOLEAN fstop     = 0;              // FIX: was missing — used in inner loop
U32     AccessCnt = 0;
U32     MemSize   = 0;
char   *file_name = "P9812d";       // driver appends .dat

int main(void)
{
    // 1. Register card
    cardID = Register_Card(PCI_9812, CardNumber);
    if (cardID < 0) {
        printf("Register_Card error=%d\n", cardID);
        return 1;
    }

    // 2. Query available DMA memory
    err = AI_InitialMemoryAllocated(cardID, &MemSize);
    if (err != NoError) {
        printf("AI_InitialMemoryAllocated error=%d\n", err);
        Release_Card(cardID);
        return 1;
    }
    printf("DMA memory available: %u KB\n", MemSize);
    if (MemSize * 1024 < (U32)(ScanCount * ADChanCount * sizeof(I16))) {
        printf("WARNING: available DMA memory (%u KB) is smaller than "
               "requested buffer (%u B). Reduce ScanCount or increase the "
               "DMA buffer size in ADLINK Device Manager.\n",
               MemSize, ScanCount * ADChanCount * (U32)sizeof(I16));
        // Continue anyway — driver may still work with a smaller effective buffer
    }

    // 3. Enable double-buffer mode  (must be before AI_9812_Config)
    err = AI_AsyncDblBufferMode(cardID, 1);
    if (err != NoError) {
        printf("AI_AsyncDblBufferMode error=%d\n", err);
        Release_Card(cardID);
        return 1;
    }

    // 4. Configure the ADC
    err = AI_9812_Config(cardID,
                         P9812_TRGMOD_SOFT,
                         P9812_TRGSRC_CH0,
                         P9812_TRGSLP_POS,
                         ClkSel,
                         TriggerLevel,
                         PostCount);
    if (err != NoError) {
        printf("AI_9812_Config error=%d\n", err);
        Release_Card(cardID);
        return 1;
    }

    // 5. Start continuous scan to file
    err = AI_ContScanChannelsToFile(cardID,
                                    LastADChan,
                                    AD_B_5_V,
                                    file_name,
                                    (U32)(ScanCount * ADChanCount),
                                    SampleRate,
                                    ASYNCH_OP);
    if (err != NoError) {
        printf("AI_ContScanChannelsToFile error=%d\n", err);
        Release_Card(cardID);
        return 1;
    }

    printf("Acquiring to %s.dat — press Ctrl-C to stop.\n", file_name);

    // 6. Poll loop: wait for each half, flush it to file
    do {
        do {
            AI_AsyncDblBufferHalfReady(cardID, &HalfReady, &fstop);
        } while (!HalfReady);

        AI_AsyncDblBufferTransfer(cardID, NULL);   // flush half → .dat
        printf(".");
        fflush(stdout);

    } while (!fstop);   // FIX: was "//here to add the condition" — stop on hardware flag

    // 7. Stop acquisition
    err = AI_AsyncClear(cardID, &AccessCnt);
    if (err != NoError) {
        printf("\nAI_AsyncClear error=%d\n", err);
    }
    printf("\nDone. %u samples transferred to %s.dat\n", AccessCnt, file_name);

    Release_Card(cardID);
    return 0;
}
