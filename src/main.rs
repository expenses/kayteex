use std::collections::HashMap;

fn main() -> std::io::Result<()> {
    #[rustfmt::skip]
    let file = &[
        // Header
        0xAB, 0x4B, 0x54, 0x58, // first four bytes of Byte[12] identifier
        0x20, 0x32, 0x30, 0xBB, // next four bytes of Byte[12] identifier
        0x0D, 0x0A, 0x1A, 0x0A, // final four bytes of Byte[12] identifier
        0x00, 0x00, 0x00, 0x00, // UInt32 vkFormat = VK_FORMAT_UNDEFINED (0)
        0x01, 0x00, 0x00, 0x00, // UInt32 typeSize = 1
        0x08, 0x00, 0x00, 0x00, // UInt32 pixelWidth = 8
        0x08, 0x00, 0x00, 0x00, // UInt32 pixelHeight = 8
        0x00, 0x00, 0x00, 0x00, // UInt32 pixelDepth = 0
        0x00, 0x00, 0x00, 0x00, // UInt32 layerCount = 0
        0x01, 0x00, 0x00, 0x00, // UInt32 faceCount = 0
        0x01, 0x00, 0x00, 0x00, // UInt32 levelCount = 0
        0x01, 0x00, 0x00, 0x00, // UInt32 supercompressionScheme = 1 (BASISLZ)
        // Index
        0x68, 0x00, 0x00, 0x00, // Uint32 dfdByteOffset = 0x00000068
        0x5C, 0x00, 0x00, 0x00, // UInt32 dfdByteSize = 0x0000005C
        0xC4, 0x00, 0x00, 0x00, // UInt32 kvdByteOffset = 0x000000C4
        0x58, 0x00, 0x00, 0x00, // UInt32 kvdByteLength = 0x00000058
        0x20, 0x01, 0x00, 0x00, // UInt64 sgdByteOffset = 0x0000000000000120
        0x00, 0x00, 0x00, 0x00,
        0x90, 0x00, 0x00, 0x00, // UInt64 sgdByteLength = 0x0000000000000090
        0x00, 0x00, 0x00, 0x00,
        // Level Index
        0xB0, 0x01, 0x00, 0x00, // UInt64 level[0].byteOffset = 0x00000000000001B0
        0x00, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, // UInt64 level[0].byteLength = 0x0000000000000003
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, // UInt64 level[0].uncompressedByteLength = 0
        0x00, 0x00, 0x00, 0x00,
        // DFD
        0x3C, 0x00, 0x00, 0x00, // UInt32 dfdTotalSize = 0x3C (60)
        0x00, 0x00, 0x00, 0x00, // vendorId = 0 (17 bits), descriptorType = 0
        0x02, 0x00, 0x38, 0x00, // versionNumber = 2, descriptorBlockSize = 0x38 (56)
        0xA3, 0x01, 0x02, 0x00, // colorModel = ETC1S (163), primaries = BT709 (1)
                                // transferFunction = SRGB (2), flags = 0
        0x03, 0x03, 0x00, 0x00, // texelBlockDimension[[0-3] = 3, 3, 0, 0
        0x00, 0x00, 0x00, 0x00, // bytesPlane[0-3] = 0
        0x00, 0x00, 0x00, 0x00, // bytesPlane[4-7] = 0
        // DFD sample information, sample 0
        0x00, 0x00, 0x3F, 0x00, // bitOffset = 0 bitLength = 0x3F (63),
                                // channelType = RGB (0), qualifiers = 0
        0x00, 0x00, 0x00, 0x00, // samplePosition[0-3] = 0
        0x00, 0x00, 0x00, 0x00, // sampleLower = 0
        0xFF, 0xFF, 0xFF, 0xFF, // sampleUpper = 0xFFFFFFFF (UINT_MAX)
        // Sample 1
        0x40, 0x00, 0x3F, 0x0F, // bitOffset = 0x40 (64) bitLength = 0x3F (63),
                                // channelType = AAA (0x0F), qualifiers = 0
        0x00, 0x00, 0x00, 0x00, // samplePosition[0-3] = 0
        0x00, 0x00, 0x00, 0x00, // sampleLower = 0
        0xFF, 0xFF, 0xFF, 0xFF, // sampleUpper = 0xFFFFFFFF (UINT_MAX)
        // Key/Value Data
        0x12, 0x00, 0x00, 0x00, // keyAndValueByteLength = 18 (0x12)
        0x4B, 0x54, 0x58, 0x6F, // KTXo
        0x72, 0x69, 0x65, 0x6E, // rien
        0x74, 0x61, 0x74, 0x69, // tati
        0x6F, 0x6E, 0x00, 0x72, // on NUL r
        0x64, 0x00, 0x00, 0x00, // d  <3 bytes of valuePadding>
        0x3B, 0x00, 0x00, 0x00, // keyAndValueByteLength = 59 (0x3B)
        0x4B, 0x54, 0x58, 0x77, // KTXw
        0x72, 0x69, 0x74, 0x65, // rite
        0x72, 0x00, 0x74, 0x6F, // r NUL to
        0x6B, 0x74, 0x78, 0x20, // ktx SPACE
        0x76, 0x34, 0x2E, 0x30, // v4.0
        0x2E, 0x5F, 0x5F, 0x64, // .__d
        0x65, 0x66, 0x61, 0x75, // efau
        0x6C, 0x74, 0x5F, 0x5F, // lt__
        0x20, 0x2F, 0x20, 0x6C, // SPACE / SPACE l
        0x69, 0x62, 0x6B, 0x74, // ibkt
        0x78, 0x20, 0x76, 0x34, // x v4
        0x2E, 0x30, 0x2E, 0x5F, // .0._
        0x5F, 0x64, 0x65, 0x66, // _def
        0x61, 0x75, 0x6C, 0x74, // ault
        0x5F, 0x5F, 0x00, 0x00, // __ <2 bytes of valuePadding>
        0x00, 0x00, 0x00, 0x00, // 4 bytes of padding.
        // Supercompression Global Data
        0x02, 0x00, 0x02, 0x00, // UInt16 endpointCount = 2, UInt16 selectorCount = 2
        0x2D, 0x00, 0x00, 0x00, // UInt32 endpointsByteLength = 0x2D
        0x09, 0x00, 0x00, 0x00, // UInt32 selectorsByteLength = 0x09
        0x2E, 0x00, 0x00, 0x00, // Uint32 tablesByteLength = 0x2E
        0x00, 0x00, 0x00, 0x00, // Uint32 extendedByteLength = 0
        // imageDesc[0]
        0x00, 0x00, 0x00, 0x00, // UInt32 flags = 0
        0x00, 0x00, 0x00, 0x00, // UInt32 rgbSliceByteOffset = 0
        0x02, 0x00, 0x00, 0x00, // UInt32 rgbSliceByteLength = 2
        0x02, 0x00, 0x00, 0x00, // UInt32 alphaSliceByteOffset = 0x02
        0x01, 0x00, 0x00, 0x00, // UInt32 alphaSliceByteLength = 1
        // endpointsData
        0x01, 0xC0, 0x04, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x02, 0x04, 0x98,
        0x1B, 0x20, 0x00, 0x00,
        0x00, 0x08, 0xC3, 0x36,
        0x91, 0x3E, 0x91, 0x00,
        0x60, 0x02, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x81, 0x00, 0x4C, 0x01,
        0x10, 0x00, 0x00, 0x00,
        0x00, 0x20, 0x59, 0xC0,
        0x3D,
        // selectorsData
            0x54, 0x55, 0x55,
        0x55, 0xAD, 0xAA, 0xAA,
        0xAA, 0x02,
        // tablesData
                    0x14, 0xC0,
        0x44, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x12,
        0x41, 0x00, 0x98, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x40, 0x18, 0x02,
        0xA2, 0x04, 0x0C, 0x00,
        0x00, 0x00, 0x83, 0x76,
        0x7B, 0x49, 0x04, 0xA2,
        0x20, 0x00, 0x4C, 0x00,
        0x08, 0x00, 0x00, 0x00,
        0x00, 0x20, 0x02, 0x01,
        // Level 0 image data
        0x4E, 0x0E, 0x04
    ][..];

    for file in std::env::args().skip(1) {
        dbg!(&file);
        let file = std::fs::File::open(file).unwrap();

        let ktx2 = Reader::new(file)?;

        dbg!(&ktx2.header);

        /*for (compressed, decompressed) in ktx2.compressed_images.iter().zip(ktx2.images_decompressed()) {
            //dbg!((compressed.len(), decompressed.unwrap().len()));
        }*/

        dbg!(&ktx2.key_values);
    }


    Ok(())
}

#[derive(Debug)]
struct Header {
    vk_format: u32,
    type_size: u32,
    pixel_width: u32,
    pixel_height: u32,
    pixel_depth: u32,
    layer_count: u32,
    face_count: u32,
    level_count: u32,
    supercompression_scheme: SupercompressionScheme,
}

#[derive(Debug)]
struct Index {
    dfd_byte_offset: u32,
    dfd_byte_length: u32,
    kvd_byte_offset: u32,
    kvd_byte_length: u32,
    sgd_byte_offset: u64,
    sgd_byte_length: u64,
}

#[derive(Debug)]
struct Level {
    byte_offset: u64,
    byte_length: u64,
    uncompressed_byte_length: u64,
}

struct Reader<T> {
    read_bytes: usize,
    inner: T,
}

impl<T: std::io::Read> Reader<T> {
    fn new(inner_reader: T) -> std::io::Result<Ktx2> {
        let mut this = Self {
            read_bytes: 0,
            inner: inner_reader,
        };

        let magic = [
            b'\xAB', b'K', b'T', b'X', b' ', b'2', b'0', b'\xBB', b'\r', b'\n', b'\x1A', b'\n'
        ];

        assert_eq!(this.read_array::<12>()?, magic);

        let header = Header {
            vk_format: this.read_u32()?,
            type_size: this.read_u32()?,
            pixel_width: this.read_u32()?,
            pixel_height: this.read_u32()?,
            pixel_depth: this.read_u32()?,
            layer_count: this.read_u32()?,
            face_count: this.read_u32()?,
            level_count: this.read_u32()?,
            supercompression_scheme: SupercompressionScheme::parse(this.read_u32()?),
        };

        let index = Index {
            dfd_byte_offset: this.read_u32()?,
            dfd_byte_length: this.read_u32()?,
            kvd_byte_offset: this.read_u32()?,
            kvd_byte_length: this.read_u32()?,
            sgd_byte_offset: this.read_u64()?,
            sgd_byte_length: this.read_u64()?,
        };

        let levels = (0..header.level_count.max(1))
            .map(|_| {
                Ok(Level {
                    byte_offset: this.read_u64()?,
                    byte_length: this.read_u64()?,
                    uncompressed_byte_length: this.read_u64()?,
                })
            })
            .collect::<std::io::Result<Vec<_>>>()?;

        let dfd_start = this.read_bytes;

        let _dfd_total_size = this.read_u32()?;

        let descriptor_block_header = DescriptorBlockHeader {
            vendor_id_and_descriptor_type: this.read_u32()?,
            version_number: this.read_u16()?,
            descriptor_block_size: this.read_u16()?,
        };

        let block = DescriptorBlock {
            color_model: ColorModel::parse(this.read_u8()?).unwrap(),
            transfer_function: TransferFunction::parse(this.read_u8()?).unwrap(),
            color_primaries: this.read_u8()?,
            flags: DescriptorBlockFlags { bits: this.read_u8()?},
            texel_block_dimensions: this.read_array()?,
            bytes_plane: this.read_array()?,
        };

        dbg!(descriptor_block_header, block);

        //dbg!(descriptor_block_header, block);

        let sample = SampleInformation {
            bit_offset: this.read_u16()?,
            bit_length: this.read_u8()?,
            channel_type_and_qualifiers: this.read_u8()?,
            sample_position: this.read_array()?,
            sample_upper: this.read_u32()?,
            sample_lower: this.read_u32()?,
        };

        /*
        let sample2 = SampleInformation {
            bit_offset: this.read_u16()?,
            bit_length: this.read_u8()?,
            channel_type_and_qualifiers: this.read_u8()?,
            sample_position: this.read_array()?,
            sample_upper: this.read_u32()?,
            sample_lower: this.read_u32()?,
        };
        */

        let kv_start_bytes = this.read_bytes;

        let mut key_values = HashMap::new();

        while this.read_bytes - kv_start_bytes < index.kvd_byte_length as usize {
            let (key, value) = this.read_key_value_pair()?;
            key_values.insert(key, value);
        }

        if index.sgd_byte_length > 0 {
            this.align(8)?;
        }

        if let SupercompressionScheme::BasisLZ = header.supercompression_scheme {
            let global_data_header = BasisLZGlobalDataHeader {
                endpoint_count: this.read_u16()?,
                selector_count: this.read_u16()?,
                endpoints_byte_length: this.read_u32()?,
                selectors_byte_length: this.read_u32()?,
                tables_byte_length: this.read_u32()?,
                extended_byte_length: this.read_u32()?,
            };

            let image_desc = BasisLZGlobalDataImageDesc {
                flags: this.read_u32()?,
                rgb_slice_byte_offset: this.read_u32()?,
                rgb_slice_byte_length: this.read_u32()?,
                alpha_slice_byte_offset: this.read_u32()?,
                alpha_slice_byte_length: this.read_u32()?,
            };

            let endpoints_data = this.read_vec(global_data_header.endpoints_byte_length as usize)?;
            let selectors_data = this.read_vec(global_data_header.selectors_byte_length as usize)?;
            let tables_data = this.read_vec(global_data_header.tables_byte_length as usize)?;

            dbg!(&global_data_header, image_desc);
        }

        let start_of_levels = this.read_bytes;

        let mut level_data_length = 0;

        for level in &levels {
            level_data_length = level_data_length.max((level.byte_length + level.byte_offset) as usize - this.read_bytes);
        }

        let compressed_image_data = this.read_vec(level_data_length)?;
        let mut compressed_images = Vec::with_capacity(levels.len());

        for level in &levels {
            let offset = level.byte_offset as usize - start_of_levels;

            compressed_images.push(Vec::from(&compressed_image_data[
                offset ..
                offset + level.byte_length as usize
            ]));
        }

        Ok(Ktx2 {
            header,
            compressed_images,
            key_values,
        })
    }

    fn read_vec(&mut self, len: usize) -> std::io::Result<Vec<u8>> {
        let mut vec = vec![0; len];
        self.inner.read_exact(&mut vec)?;
        self.read_bytes += len;
        Ok(vec)
    }

    fn read_array<const LEN: usize>(&mut self) -> std::io::Result<[u8; LEN]> {
        let mut array = [0; LEN];
        self.inner.read_exact(&mut array)?;
        self.read_bytes += LEN;
        Ok(array)
    }

    fn read_u8(&mut self) -> std::io::Result<u8> {
        Ok(u8::from_le_bytes(self.read_array()?))
    }

    fn read_u16(&mut self) -> std::io::Result<u16> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> std::io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> std::io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn align(&mut self, bytes: usize) -> std::io::Result<()> {
        let remainder = self.read_bytes % bytes;

        if remainder > 0 {
            let to_read = bytes - remainder;
            self.read_vec(to_read)?;
        }

        Ok(())
    }

    fn read_key_value_pair(&mut self) -> std::io::Result<(String, String)> {
        let length = self.read_u32()?;
        let key_and_value = self.read_vec(length as usize)?;

        let key_end_index = key_and_value.iter().position(|&c| c == b'\0').unwrap();
        let key = unsafe {
            std::str::from_utf8_unchecked(&key_and_value[..key_end_index as usize])
        };
        let value = unsafe {
            std::str::from_utf8_unchecked(&key_and_value[key_end_index as usize + 1 .. key_and_value.len() - 1])
        };

        self.align(4)?;

        Ok((
            key.into(), value.into(),
        ))
    }
}

struct Ktx2 {
    header: Header,
    compressed_images: Vec<Vec<u8>>,
    key_values: HashMap<String, String>,
}

impl Ktx2 {
    fn images_decompressed(&self) -> impl Iterator<Item = anyhow::Result<Vec<u8>>> + '_ {
        self.compressed_images.iter()
            .map(|image| {
                match &self.header.supercompression_scheme {
                    SupercompressionScheme::None => Ok(image.clone()),
                    SupercompressionScheme::Zstandard => {
                        Ok(zstd::stream::decode_all(&image[..])?)
                    },
                    scheme => panic!("Not supported: {:?}", scheme)
                }
            })
    }
}

#[derive(Debug)]
struct DescriptorBlockHeader {
    // vendor id is 17 bits
    // https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html#_anchor_id_descriptorblock_xreflabel_descriptorblock_descriptor_block
    vendor_id_and_descriptor_type: u32,
    version_number: u16,
    descriptor_block_size: u16,
}

#[derive(Debug)]
struct DescriptorBlock {
    color_model: ColorModel,
    transfer_function: TransferFunction,
    color_primaries: u8,
    flags: DescriptorBlockFlags,
    texel_block_dimensions: [u8; 4],
    bytes_plane: [u8; 8],
}

#[derive(Debug)]
enum SupercompressionScheme {
    None = 0,
    BasisLZ = 1,
    Zstandard = 2,
    ZLIB = 3,
    Reserved = 4,
}

impl SupercompressionScheme {
    fn parse(value: u32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::BasisLZ,
            2 => Self::Zstandard,
            3 => Self::ZLIB,
            _ => Self::Reserved,
        }
    }
}

bitflags::bitflags! {
    struct DescriptorBlockFlags: u8 {
        const ALPHA_STRAIGHT = 0b0;
        const ALPHA_PREMULTIPLIED = 0b1;
    }
}

#[derive(Debug)]
struct SampleInformation {
    bit_offset: u16,
    bit_length: u8,
    channel_type_and_qualifiers: u8,
    sample_position: [u8; 4],
    sample_upper: u32,
    sample_lower: u32,
}

#[derive(Debug)]
struct BasisLZGlobalDataHeader {
    endpoint_count: u16,
    selector_count: u16,
    endpoints_byte_length: u32,
    selectors_byte_length: u32,
    tables_byte_length: u32,
    extended_byte_length: u32,
}

#[derive(Debug)]
struct BasisLZGlobalDataImageDesc {
    flags: u32,
    rgb_slice_byte_offset: u32,
    rgb_slice_byte_length: u32,
    alpha_slice_byte_offset: u32,
    alpha_slice_byte_length: u32,
}

#[derive(Debug)]
enum ColorModel {
    ETC1S = 163,
    Uastc = 166,
}

impl ColorModel {
    fn parse(value: u8) -> Option<Self> {
        match value {
            163 => Some(Self::ETC1S),
            166 => Some(Self::Uastc),
            _ => {
                dbg!(value);
                None
            },
        }
    }
}

#[derive(Debug)]
enum TransferFunction {
    Unspecified = 0,
    Linear = 1,
    SRGB = 2,
}

impl TransferFunction {
    fn parse(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unspecified),
            1 => Some(Self::Linear),
            2 => Some(Self::SRGB),
            _ => None
        }
    }
}
