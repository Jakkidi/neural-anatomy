// A few GPUs to run the GPU view against: the sizes that decide how attention
// is laid out on them. Published figures, rounded; a real kernel gets somewhat
// less than the peak bandwidth.

export const DEVICES = [
  {
    id: "a100",
    name: "NVIDIA A100 (40 GB)",
    sms: 108,
    sram: 192 * 1024, sramNote: "192 KB of L1 and shared memory per SM, up to 164 KB of it shared",
    memory: "40 GB HBM2", memName: "HBM",
    bandwidth: 1.555e12,
    sramBandwidth: "about 19 TB/s, all SMs together (the FlashAttention paper's estimate)",
  },
  {
    id: "h100",
    name: "NVIDIA H100 SXM (80 GB)",
    sms: 132,
    sram: 256 * 1024, sramNote: "256 KB of L1 and shared memory per SM, up to 228 KB of it shared",
    memory: "80 GB HBM3", memName: "HBM",
    bandwidth: 3.35e12,
  },
  {
    id: "rtx4090",
    name: "NVIDIA GeForce RTX 4090",
    sms: 128,
    sram: 128 * 1024, sramNote: "128 KB of L1 and shared memory per SM",
    memory: "24 GB GDDR6X", memName: "VRAM",
    bandwidth: 1.008e12,
  },
];

export const deviceById = (id) => DEVICES.find((d) => d.id === id) ?? DEVICES[0];
