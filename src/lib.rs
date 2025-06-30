use std::{fmt::{Debug, Display}, ops::{Index, IndexMut}, thread::current};
use std::hash::Hash;

use fnv::FnvHashSet;



pub const NULLPTR: Pointer = Pointer{pointer: usize::MAX};

pub trait Null: PartialEq + Sized {
    fn null() -> Self;

    fn is_null(&self) -> bool {
        self == &Self::null()
    }
}


impl Null for i32 {
    fn null() -> i32 {
        i32::MAX
    }
}

impl Null for f32 {
    fn null() -> f32 {
        std::f32::NAN
    }
}

impl Null for u32 {
    fn null() -> u32 {
        u32::MAX
    }
}


impl Null for usize {
    fn null() -> usize {
        usize::MAX
    }
}



#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Pointer {
    pub pointer: usize,
}

impl Null for Pointer {
    fn null() -> Self {
        NULLPTR
    }
}

impl Display for Pointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_null() {
            write!(f, "NULL")
        } else {
            write!(f, "ptr({})", self.pointer)
        }
    }
}

#[inline]
pub fn ptr(u: usize) -> Pointer {
    Pointer{pointer: u}
}



#[derive(Clone, Debug)]
pub struct FreeListVec<T: Null> {
    list: Vec<T>,
    free_list: FnvHashSet<usize>,
}

impl<T: Null + Clone> FreeListVec<T> {

    pub fn new() -> FreeListVec<T> {
        FreeListVec {
            list: Vec::new(),
            free_list: FnvHashSet::default(),
        }
    }

    pub fn add(&mut self, t: T) -> Pointer {
        match pop_from_hashset(&mut self.free_list) {
            Some(index) => {self.list[index] = t; return ptr(index)},
            None => {self.list.push(t); return ptr(self.list.len() - 1)},
        }
    }

    pub fn remove(&mut self, index: Pointer) -> T {
        if self.free_list.contains(&index.pointer) {
            panic!()
        } else  {
            let res = self.list[index.pointer].clone();
            self.list[index.pointer] = T::null();
            self.free_list.insert(index.pointer);
            return res
        }
    }

    pub fn split_at_mut(&mut self, index: usize) -> (SplitFreeList<T>, SplitFreeList<T>) {
        let (left_slice, right_slice) = self.list.split_at_mut(index);
        let left_list = SplitFreeList {
            slice: left_slice,
            free_list: &self.free_list,
        };

        let right_list = SplitFreeList {
            slice: right_slice,
            free_list: &self.free_list,
        };

        (left_list, right_list)
    }
}

pub struct SplitFreeList<'a, T> {
    pub slice: &'a mut [T],
    free_list: &'a FnvHashSet<usize>,
}

impl<'a, T: Null + Clone> Index<Pointer> for SplitFreeList<'a, T> {
    type Output = T;

    fn index(&self, index: Pointer) -> &Self::Output {
        if self.free_list.contains(&index.pointer) {
            panic!("Tried to access a freed value with index: {}", index.pointer)
        }
        &self.slice[index.pointer]
    }
}

impl<'a, T: Null + Clone> IndexMut<Pointer> for SplitFreeList<'a, T> {

    fn index_mut(&mut self, index: Pointer) -> &mut Self::Output {
        if self.free_list.contains(&index.pointer) {
            panic!("Tried to access a freed value with index: {}", index.pointer)
        }
        &mut self.slice[index.pointer]
    }
}

impl<T: Null + Clone> Index<Pointer> for FreeListVec<T> {
    type Output = T;

    fn index(&self, index: Pointer) -> &Self::Output {
        if self.free_list.contains(&index.pointer) {
            panic!("Tried to access a freed value with index: {}", index.pointer)
        }
        &self.list[index.pointer]
    }
}

impl<T: Null + Clone> IndexMut<Pointer> for FreeListVec<T> {

    fn index_mut(&mut self, index: Pointer) -> &mut Self::Output {
        if self.free_list.contains(&index.pointer) {
            panic!("Tried to access a freed value with index: {}", index.pointer)
        }
        &mut self.list[index.pointer]
    }
}

impl<'a, T: Null> IntoIterator for &'a FreeListVec<T> {
    type Item = &'a T;
    type IntoIter = FreeListIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        FreeListIter {
            items: &self.list,
            index: 0,
        }
    }
}


pub struct FreeListIter<'a, T: Null> {
    items: &'a [T],
    index: usize,
}

impl<'a, T: Null> Iterator for FreeListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.items.len() {
            let item = &self.items[self.index];
            self.index += 1;

            if !item.is_null() {
                return Some(item);
            }
        }
        None
    }
}

pub fn extend_zeroes(vec: &mut Vec<u8>, n: usize) {
    vec.resize(vec.len() + n, 0);
}


pub struct Hallocator {
    pub buffer: Vec<u8>,
    block_size: usize,
    tail: usize,
    free_list: FnvHashSet<usize>,
}

impl Hallocator {
    pub fn new(block_size: usize) -> Hallocator {
        Hallocator {
            buffer: Vec::with_capacity(block_size * 64),
            block_size,
            tail: 0,
            free_list: FnvHashSet::default(),
        }
    }

    pub fn alloc(&mut self) -> Pointer {
        
        match pop_from_hashset(&mut self.free_list) {
            Some(pointer) => {
                Pointer{pointer}
            },
            None => {
                let result = self.tail;
                extend_zeroes(&mut self.buffer, self.block_size);
                self.tail += self.block_size;
                Pointer{pointer: result}
            },
        }
    }

    pub fn free(&mut self, pointer: usize) -> Result<(), String> {
        match self.free_list.insert(pointer) {
            true => (),
            false => return Err(format!("Attempting to double free a pointer. Pointer address: {}", pointer as usize)),
        }
        let row_pointer = &self.buffer[pointer..pointer + self.block_size].as_mut_ptr();
        unsafe { row_pointer.write_bytes(0, self.block_size) };

        Ok(())
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[inline]
    pub fn get_block(&self, pointer: Pointer) -> &[u8] {
        let pointer = pointer.pointer;
        &self.buffer[pointer..pointer+self.block_size]
    }

    #[inline]
    pub fn get_block_mut(&mut self, pointer: Pointer) -> &mut [u8] {
        let pointer = pointer.pointer;

        &mut self.buffer[pointer..pointer+self.block_size]
    }

    #[inline]
    pub fn read_i32(&self, pointer: Pointer, offset: usize) -> i32 {
        let pointer = pointer.pointer;

        if offset > self.block_size - 4 {
            panic!("Trying to read out of bounds memory")
        }
        unsafe { *(self.get_block(ptr(pointer+offset)).as_ptr() as *const i32) }
    }

    #[inline]
    pub fn read_u64(&self, pointer: Pointer, offset: usize) -> u64 {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 8 {
            panic!("Trying to read out of bounds memory")
        }
        unsafe { *(self.get_block(ptr(pointer+offset)).as_ptr() as *const u64) }
    }

    #[inline]
    pub fn read_f32(&self, pointer: Pointer, offset: usize) -> f32 {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 4 {
            panic!("Trying to read out of bounds memory")
        }
        unsafe { *(self.get_block(ptr(pointer+offset)).as_ptr() as *const f32) }
    }

    #[inline]
    pub fn write_i32(&mut self, pointer: Pointer, offset: usize, value: i32) {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 4 {
            panic!("Trying to write out of bounds memory")
        }
        unsafe { (self.get_block_mut(ptr(pointer+offset)).as_mut_ptr() as *mut i32).write(value) }
    }

    #[inline]
    pub fn write_u64(&mut self, pointer: Pointer, offset: usize, value: u64) {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 8 {
            panic!("Trying to write out of bounds memory")
        }
        unsafe { (self.get_block_mut(ptr(pointer+offset)).as_mut_ptr() as *mut u64).write(value) }
    }

    #[inline]
    pub fn write_f32(&mut self, pointer: Pointer, offset: usize, value: f32) {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 4 {
            panic!("Trying to write out of bounds memory")
        }
        unsafe { (self.get_block_mut(ptr(pointer+offset)).as_mut_ptr() as *mut f32).write(value) }
    }

    
}








#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct FixedList<T: Null + Clone + Copy + Debug + Ord + Eq + Sized, const N: usize> {
    list: [T ; N],
    len: usize,
}

impl<T: Null + Clone + Copy + Debug + Ord + Eq + Sized, const N: usize> FixedList<T, N> {
    pub fn new() -> FixedList<T, N> {
        FixedList {
            list: std::array::from_fn(|_| T::null()),
            len: 0,
        }
    }

    pub fn push(&mut self, t: T) -> bool {
        if self.len > self.list.len() {
            return false
        } else {
            self.list[self.len] = t;
            self.len += 1;
            return true
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            let result = self.list[self.len].clone();
            self.list[self.len] = T::null();
            self.len -= 1;
            Some(result)
        }
    }

    pub fn full(&self) -> bool {
        self.len == N
    }

    pub fn insert_at(&mut self, index: usize, value: &T) -> Result<(), String> {
        if self.full() || index > self.len {
            self.push(*value);
            return Err(format!("Tried to insert {:?} at index {} in a FixedList of len {}", value, index, self.len) )
        }

        let temp = self.list[index..].to_vec();

        self.list[index] = value.clone();
        self.len += 1;
        for i in 0..temp.len()-1 {
            self.list[index+1+i] = temp[i].clone();
        }

        Ok(())
    }

    ///Removes item at index and shifts subsequent items down
    pub fn remove(&mut self, index: usize) -> T {
        if self.len() == 0 {
            return T::null()
        }
        let t = self.list[index];
        
        for i in index..self.len() - 1 {
            self.list[i] = self.list[i + 1];
        }
        self.list[self.len() - 1] = T::null();

        self.len -= 1;

        t
    }

    pub fn sort(&mut self) {
        self.list.sort()
    }

    pub fn iter(&self) ->  std::slice::Iter<'_, T> {
        self.list[0..self.len].iter()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn search(&self, t: &T) -> usize {
        let mut i = 0;
        while i < self.len() {
            if &self.list[i] > t  {
                break
            }
            i += 1;
        }
        return i
    }

    pub fn find(&self, t: &T) -> Option<usize> {
        for i in 0..self.len() {
            if &self.list[i] == t {
                return Some(i)
            }
        }

        None
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.list.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.list.get_mut(index)
    }

    pub fn get_last_slot(&self) -> T {
        self.list[N-1].clone()
    }

    pub fn set_last_slot(&mut self, t: &T) {
        self.list[N-1] = t.clone();
    }
    
    pub fn get_last(&self) -> Option<&T> {
        self.list.get(self.len-1)
    }

    pub fn get_last_mut(&mut self) -> Option<&mut T> {
        self.list.get_mut(self.len-1)
    }

    pub fn get_end_slot(&self) -> T {
        self.list[N-1].clone()
    }

    pub fn set_end_slot(&mut self, value: T) {
        self.list[N-1] = value;
    }

    pub fn set(&mut self, index: usize, value: T) {
        self.list[index] = value;
    }

    pub fn drain(&mut self, other: &mut FixedList<T, N>) {

        let mut head = 0;
        
        while head < other.len && head < N - self.len {
            self.push(other.list[head].clone());
            head += 1;
        }

        other.len -= head;

        for i in 0..other.len {
            other.list[i] = other.list[i + head].clone();
        }

    }
}

impl<T: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized, const N: usize> Display for FixedList<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut printer = String::from("[");
        for item in &self.list[0..self.len()] {
            printer.push_str(&format!("{}, ", item));
        }
        printer.push(']');

        write!(f, "{}",printer)
    }
}

impl<T: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized, const N: usize> Index<usize> for FixedList<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.list[index]
    }
}

impl<T: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized, const N: usize> IndexMut<usize> for FixedList<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.list[index]
    }
}



pub fn pop_from_hashset<T: Eq + Hash + Clone>(set: &mut FnvHashSet<T>) -> Option<T> {
    let result = match set.iter().next() {
        Some(item) => item,
        None => return None,
    };
    let key = result.clone();

    set.take(&key)
}


///ORDER MUST BE AN EVEN NUMBER
pub const ORDER: usize = 6;
pub const ORDER_PLUS_ONE: usize = ORDER + 1;


#[derive(Clone, PartialEq, Debug)]
pub struct BPlusTreeNode<T: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display> {
    keys: FixedList<T, ORDER>,
    parent: Pointer,
    children: FixedList<Pointer, ORDER_PLUS_ONE>,
    is_leaf: bool,
}

impl<T: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display> Null for BPlusTreeNode<T> {
    fn null() -> Self {
        BPlusTreeNode::new_branch()
    }
}

impl<T: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized> Display for BPlusTreeNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_leaf {
            writeln!(f, "LEAF:\nparent: {}\nis_leaf: {}\nkeys: {}\nchildren: {}\nRight_sibling: {}", self.parent, self.is_leaf, self.keys, self.children, self.get_right_sibling_pointer())
        } else {
            writeln!(f, "BRANCH:\nparent: {}\nis_leaf: {}\nkeys: {}\nchildren: {}\n", self.parent, self.is_leaf, self.keys, self.children)

        }
    }
}


impl <T: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display + Display> BPlusTreeNode<T> {
    pub fn new(key: &T, pointer: Pointer) -> BPlusTreeNode<T> {
        let mut keys: FixedList<T, ORDER> = FixedList::new();
        keys.push(key.clone());
        let mut children = FixedList::new();
        children.push(pointer);
        BPlusTreeNode { keys, children, parent: ptr(usize::MAX), is_leaf: true }
    }

    pub fn new_branch() -> BPlusTreeNode<T> {
        BPlusTreeNode { keys: FixedList::new(), parent: NULLPTR, children: FixedList::new(), is_leaf: false }
    }

    pub fn new_leaf() -> BPlusTreeNode<T> {
        BPlusTreeNode { keys: FixedList::new(), parent: NULLPTR, children: FixedList::new(), is_leaf: true }
    }

    pub fn clear(&mut self) {
        self.children = FixedList::new();
        self.keys = FixedList::new();
    }

    fn get_right_sibling_pointer(&self) -> Pointer {
        self.children.get_end_slot()
    }

    fn set_right_sibling_pointer(&mut self, pointer: Pointer) {
        self.children.set_end_slot(pointer);
    }

}




pub struct BPlusTreeMap<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display + Display> {
    name: String,
    root_node: Pointer,
    nodes: FreeListVec<BPlusTreeNode<K>>,
}

impl<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display> Display for BPlusTreeMap<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        
        let mut printer = String::new();

        for (i, node) in self.nodes.into_iter().enumerate() {
            printer.push_str(&format!("{} - {} - \n", i, node));
        }
        
        writeln!(f, "{}", printer)
    }
}

impl<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display> BPlusTreeMap<K> {
    pub fn new(name: String) -> BPlusTreeMap<K> {
        let mut root: BPlusTreeNode<K> = BPlusTreeNode::new_branch();
        root.is_leaf = true;
        let mut nodes = FreeListVec::new();
        let root_pointer = nodes.add(root);
        BPlusTreeMap {
            name,
            root_node: root_pointer, 
            nodes,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn find_leaf(&self, key: &K) -> Pointer {

        let mut node = &self.nodes[self.root_node];
        
        let mut node_pointer = self.root_node;
        // let mut i: usize;
        while !node.is_leaf {
            let key_slot = node.keys.search(key);
            node_pointer = node.children[key_slot];
            node = &self.nodes[node_pointer];
            // i = 0;
            // while i < node.keys.len() {
            //     if key >= &node.keys[i] {
            //         i += 1;
            //     }
            //     else {
            //         break;
            //     }
            // }
            // node_pointer = node.children[i];
            // if node_pointer.is_null() {
            //     println!("{}", self);
            // }
            // node = &self.nodes[node_pointer];
        }
        node_pointer
    }

    pub fn insert(&mut self, key: &K, value: Pointer) {
        let node_pointer = self.find_leaf(key);
        
        self.insert_into_leaf(key, value, node_pointer);
    }

    fn insert_into_leaf(&mut self, key: &K, value_pointer: Pointer, target_node_pointer: Pointer) {

        let node = &mut self.nodes[target_node_pointer];
        // println!("node: {}\n{}", node_pointer, node);

        if node.keys.len() > ORDER {
            panic!()
        }

        
        if node.keys.len() == ORDER {
            
            let new_key_index = node.keys.search(key);
            let mut left_node: BPlusTreeNode<K>;
            let mut right_node: BPlusTreeNode<K>;
            if node.is_leaf {
                left_node = BPlusTreeNode::new_leaf();
                right_node = BPlusTreeNode::new_leaf();
                
            } else {
                left_node = BPlusTreeNode::new_branch();
                right_node = BPlusTreeNode::new_branch();
            }
            
            for i in 0 .. node.keys.len() {
                let k = node.keys[i];
                let p = node.children[i];
                if i < cut(ORDER) {
                    left_node.keys.push(k);
                    left_node.children.push(p);
                } else {
                    right_node.keys.push(k);
                    right_node.children.push(p);
                }
            }
            if new_key_index < cut(ORDER) {
                left_node.keys.insert_at(new_key_index, key).unwrap();
                left_node.children.insert_at(new_key_index, &value_pointer).unwrap();
            } else {
                left_node.keys.insert_at(new_key_index, key).unwrap();
                left_node.children.insert_at(new_key_index, &value_pointer).unwrap();
                
            }
            let key = node.keys[cut(ORDER)];

            let mut parent_pointer = node.parent;
            if parent_pointer == NULLPTR {
                assert!(self.root_node == target_node_pointer);
                let new_root_node: BPlusTreeNode<K> = BPlusTreeNode::new_branch();
                
                parent_pointer = self.nodes.add(new_root_node);
                self.root_node = parent_pointer;
                left_node.parent = parent_pointer;
                right_node.parent = parent_pointer;
                self.nodes.remove(target_node_pointer);
                
                let left_pointer = self.nodes.add(left_node);
                let right_pointer = self.nodes.add(right_node);

                let left_node = &mut self.nodes[left_pointer];
                left_node.set_right_sibling_pointer(right_pointer);
                
                let new_root_node = &mut self.nodes[parent_pointer];
                new_root_node.keys.push(key);
                new_root_node.children.push(left_pointer);
                new_root_node.children.push(right_pointer);
            } else {
                left_node.parent = parent_pointer;
                right_node.parent = parent_pointer;
                
                let right_pointer = self.nodes.add(right_node);
                left_node.set_right_sibling_pointer(right_pointer);
                self.nodes[target_node_pointer] = left_node;
                
                // let left_pointer = target_node_pointer;

                // let left_node = &mut self.nodes[left_pointer];
                
                // self.update_keys(parent_pointer, left_pointer, &lower_key, &upper_key);
                self.insert_into_branch(&key, right_pointer, parent_pointer);
            }
            // drop(node);
        }
    }

    fn insert_into_branch(&mut self, key: &K, value_pointer: Pointer, target_node_pointer: Pointer) {
        let node = &mut self.nodes[target_node_pointer];

        if node.keys.len() > ORDER {
            panic!()
        }

        let index = node.keys.search(key);
        node.keys.insert_at(index, key).unwrap();
        
        node.children.insert_at(index+1, &value_pointer).unwrap();

        if node.keys.len() == ORDER {
            
            let mut left_node = BPlusTreeNode::new_branch();
            let mut right_node = BPlusTreeNode::new_branch();

            let mut i = 0;
            while i < node.keys.len() {
                let k = node.keys[i];
                let p = node.children[i];
                if i < ORDER / 2 {
                    left_node.keys.push(k);
                    left_node.children.push(p);
                } else if i == ORDER / 2 {
                    left_node.children.push(p);
                } else if i > ORDER / 2 {
                    right_node.keys.push(k);
                    right_node.children.push(p);
                }

                i += 1;
            }
            let p = node.children[i];
            right_node.children.push(p);

            let key = node.keys[cut(ORDER)];

            let mut parent_pointer = node.parent;
            if parent_pointer == NULLPTR {
                assert!(self.root_node == target_node_pointer);
                let new_root_node: BPlusTreeNode<K> = BPlusTreeNode::new_branch();
                
                parent_pointer = self.nodes.add(new_root_node);
                self.root_node = parent_pointer;
                left_node.parent = parent_pointer;
                right_node.parent = parent_pointer;
                self.nodes.remove(target_node_pointer);
                
                let left_pointer = self.nodes.add(left_node);
                let right_pointer = self.nodes.add(right_node);

                let new_root_node = &mut self.nodes[parent_pointer];
                new_root_node.keys.push(key);
                new_root_node.children.push(left_pointer);
                new_root_node.children.push(right_pointer);
            } else {
                left_node.parent = parent_pointer;
                right_node.parent = parent_pointer;
                self.nodes[target_node_pointer] = left_node;
                
                let _left_pointer = target_node_pointer;
                let right_pointer = self.nodes.add(right_node);

                // self.update_keys(parent_pointer, left_pointer, &lower_key, &upper_key);
                self.insert_into_branch(&key, right_pointer, parent_pointer);
            }
        }
    }

    pub fn get_value(&self, key: &K) -> Pointer {
        let node = self.find_leaf(key);
        if node.is_null() {
            return NULLPTR
        }
        let node = &self.nodes[node];
        
        match node.keys.find(key) {
            Some(index) => {
                return node.children.get(index).unwrap().clone();
            },
            None => return NULLPTR,
        }

    }

    fn get_left_sibling_pointer(&self, leaf_node: &BPlusTreeNode<K>) -> Pointer {
        
        let parent_node = &self.nodes[leaf_node.parent];
        let leaf_key = leaf_node.keys[0];
        let key_index = parent_node.keys.search(&leaf_key);

        if key_index == 0 {
            println!("{}", self);
        }

        return parent_node.children[key_index]
        
    }

    pub fn delete_key(&mut self, key: &K) -> Result<Pointer, String> {
        let mut current_node_pointer = self.find_leaf(key);
        if current_node_pointer.is_null() {
            return Err(format!("Key: '{:?}' does not exist in table: '{}'", key, self.name))
        }

        let current_node = &mut self.nodes[current_node_pointer];
        let key_index = match current_node.keys.find(key) {
            Some(index) => index,
            None => {
                println!("{}", self);
                panic!()
            }
        };
        current_node.keys.remove(key_index);
        let return_pointer = current_node.children.remove(key_index);

        if self.root_node == current_node_pointer {
            return Ok(return_pointer)
        } else if current_node.keys.len() < cut(ORDER) {
            self.rebalance_node(current_node_pointer);
        }
        Ok(return_pointer)
    }


    fn rebalance_node(&mut self, target_node_pointer: Pointer) {

        let current_node = &self.nodes[target_node_pointer];

        let right_sibling_pointer = current_node.get_right_sibling_pointer();

        if right_sibling_pointer.is_null() {
            // WHAT IF WE ARE IN THE RIGHTMOST NODE!!!
            if current_node.parent.is_null() {
                return
            } else {
                let parent = &self.nodes[current_node.parent];
                let left_sibling_index = match parent.children.find(&target_node_pointer) {
                    Some(index) => index - 1,
                    None => {
                        println!("{}", self);
                        panic!();
                    }
                };
                let left_sibling_pointer = parent.children[left_sibling_index];
                
            }

        } else {
            let right_sibling = &self.nodes[right_sibling_pointer];
            let right_sibling_parent = right_sibling.parent;
            if right_sibling.keys.len() == cut(ORDER) {
                let mut temp_keys = FixedList::new();
                let mut temp_children = FixedList::new();
                let right_sibling = &mut self.nodes[right_sibling_pointer];
                temp_keys.drain(&mut right_sibling.keys);
                temp_children.drain(&mut right_sibling.children);

                let current_node = &mut self.nodes[target_node_pointer];
                current_node.keys.drain(&mut temp_keys);
                current_node.children.drain(&mut temp_children);
                self.nodes.remove(right_sibling_pointer);

                let right_parent = &mut self.nodes[right_sibling_parent];
                let right_pointer_index = right_parent.children.find(&right_sibling_pointer).unwrap();
                right_parent.children.remove(right_pointer_index);
                
                if right_parent.keys.len() < cut(ORDER) {
                    self.rebalance_node(right_sibling_parent);
                } else {
                    right_parent.keys.remove(right_pointer_index - 1);
                }

            } else /* if right_sibling.keys.len() > cut(Order) */ {
                let mut temp_keys = Vec::new();
                let mut temp_children = Vec::new();

                for i in 0..current_node.keys.len() {
                    temp_keys.push(current_node.keys[i]);
                }
                for i in 0..right_sibling.keys.len() {
                    temp_keys.push(right_sibling.keys[i]);
                }

                for i in 0..current_node.children.len() {
                    temp_children.push(current_node.children[i]);
                }
                for i in 0..right_sibling.children.len() {
                    temp_children.push(right_sibling.children[i]);
                }

                let current_node = &mut self.nodes[target_node_pointer];
                for i in 0..temp_keys.len()/2 {
                    current_node.keys[i] = temp_keys[i];
                }
                let right_sibling = &mut self.nodes[right_sibling_pointer];
                for i in temp_keys.len()/2 .. temp_keys.len() {
                    right_sibling.keys[i] = temp_keys[i];
                }

                let current_node = &mut self.nodes[target_node_pointer];
                for i in 0..temp_children.len()/2 {
                    current_node.children[i] = temp_children[i];
                }
                let right_sibling = &mut self.nodes[right_sibling_pointer];
                for i in temp_children.len()/2 .. temp_children.len() {
                    right_sibling.children[i] = temp_children[i];
                }

                let new_key = right_sibling.keys[0];
                let right_parent_pointer = right_sibling.parent;
                let right_parent = &mut self.nodes[right_parent_pointer];
                let right_index = right_parent.children.find(&right_sibling_pointer).unwrap();
                right_parent.keys[right_index - 1] = new_key;
            }
        }
    }


    // pub fn delete_key(&mut self, key: &K) -> Result<(), String> {
    //     let mut current_node_pointer = self.find_leaf(key);
    //     if current_node_pointer.is_null() {
    //         return Err( format!("Key: '{:?}' does not exist in table: '{}'", key, self.name) )
    //     }

    //     let current_node = &mut self.nodes[current_node_pointer];
    //     let key_index = match current_node.keys.find(key) {
    //         Some(index) => index,
    //         None => {
    //             println!("{}", self);
    //             println!("Couldn't find key: '{}' in node: '{}'", key, current_node_pointer);
    //             panic!()
    //         }
    //     };
    //     current_node.keys.remove(key_index);
    //     current_node.children.remove(key_index);
        
    //     if current_node.parent.is_null() {
    //         return Ok(())
    //     }

    //     let mut num_keys = current_node.keys.len();
    //     while num_keys < cut(ORDER) {
    //         println!("num_keys: {}", num_keys);
    //         let current_node = &self.nodes[current_node_pointer];
    //         if current_node.parent.is_null() {
    //             return Ok(())
    //         }
    //         let mut right_sibling_pointer = current_node.get_right_sibling_pointer();
    //         if right_sibling_pointer.is_null() {
    //             /*WHAT IF WE HAVE THE RIGHTMOST NODE */

    //             let left_sibling_pointer = self.get_left_sibling_pointer(current_node);
    //             if left_sibling_pointer.is_null() {
    //                 panic!("If the parent is not null but both the left and right sibling pointers are null then the tree is broken")
    //             }
    //             right_sibling_pointer = current_node_pointer;
    //             current_node_pointer = left_sibling_pointer;

    //         }

    //         let right_sibling = &mut self.nodes[right_sibling_pointer];
    //         let mut temp_keys = FixedList::new();
    //         let mut temp_children = FixedList::new();
            
    //         if right_sibling.keys.len() == cut(ORDER) {
    //             temp_keys.drain(&mut right_sibling.keys);
    //             temp_children.drain(&mut right_sibling.children);
                
    //             let right_parent_pointer = right_sibling.parent;
    //             let right_sibling_right_sibling = right_sibling.get_right_sibling_pointer();
    //             let current_node = &mut self.nodes[current_node_pointer];
    //             current_node.keys.drain(&mut temp_keys);
    //             current_node.children.drain(&mut temp_children);
                
    //             current_node.set_right_sibling_pointer(right_sibling_right_sibling);

    //             self.nodes.remove(right_sibling_pointer);
    //             let right_parent = &mut self.nodes[right_parent_pointer];
    //             let right_index = right_parent.children.find(&right_sibling_pointer).unwrap();

    //             right_parent.keys.remove(right_index-1);
    //             right_parent.children.remove(right_index);
    //             num_keys = right_parent.keys.len();
    //             current_node_pointer = right_parent_pointer;

    //         } else {

    //             let right_sibling = &mut self.nodes[right_sibling_pointer];
    //             let parent_node_pointer = right_sibling.parent;

    //             let temp_key = right_sibling.keys.remove(0);
    //             let temp_child = right_sibling.children.remove(0);
    //             let new_key = right_sibling.keys.get(0).unwrap().clone();
                
    //             let current_node = &mut self.nodes[current_node_pointer];

    //             current_node.keys.push(temp_key);
    //             current_node.children.push(temp_child);


    //             let parent_node = &mut self.nodes[parent_node_pointer];
    //             let key_index = parent_node.children.find(&right_sibling_pointer).unwrap() - 1;
                
    //             *parent_node.keys.get_mut(key_index).unwrap() = new_key;

    //         }
    //     }

    //     Ok(())
    // }


}

pub fn check_tree_height<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display>(tree: &BPlusTreeMap<K>) -> (bool, String) {

    let mut node = &tree.nodes[tree.root_node];
        
    let mut node_pointer: Pointer;
    let mut i = 0;
    while !node.is_leaf {
        i += 1;
        node_pointer = node.children[0];
        node = &tree.nodes[node_pointer];
    }

    let leftmost_node = node;
    let mut node_pointer = leftmost_node.get_right_sibling_pointer();
    
    while !node_pointer.is_null() {
        node = &tree.nodes[node_pointer];
        node_pointer = node.get_right_sibling_pointer();
        let mut j = 0;
        let mut backtrack_node = node;
        while !backtrack_node.parent.is_null() {
            backtrack_node = &tree.nodes[backtrack_node.parent];
            j += 1;
        }
        if j != i {
            return (false, format!("Node: '{}' is at height '{}' but the leftmost node is of height '{}'", node_pointer, j, i))
        }
    }

    (true, "ALL GOOD".to_owned())
}

pub fn check_tree_ordering<K: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized>(tree: &BPlusTreeMap<K>) -> (bool, String) {

    let mut node = &tree.nodes[tree.root_node];

    let mut node_pointer = NULLPTR;
    while !node.is_leaf {
        node_pointer = node.children[0];
        node = &tree.nodes[node_pointer];
    }

    let mut last_key = node.keys[0];
    while !node_pointer.is_null() {
        node = &tree.nodes[node_pointer];
        for key in node.keys.iter() {
            if &last_key > key {
                return (false, format!("Found out of order key in node: {}. Key '{}' is larger than key: '{}'", node_pointer, last_key, key))
            } else {
                last_key = key.clone();
            }
        }
        node_pointer = node.get_right_sibling_pointer();
    }

    (true, "ALL GOOD".to_owned())
}

pub fn check_tree_leafpairs(tree: &BPlusTreeMap<u32>) -> (bool, String) {

    let mut node = &tree.nodes[tree.root_node];

    let mut node_pointer = NULLPTR;
    while !node.is_leaf {
        node_pointer = node.children[0];
        node = &tree.nodes[node_pointer];
    }

    while !node_pointer.is_null() {
        node = &tree.nodes[node_pointer];
        for i in 0..node.keys.len() {
            let key = node.keys[i];
            let value = node.children[i];
            if value != ptr(key as usize) {
                return (false, format!("In node: '{}' - Key '{}' points to Pointer '{}'", node_pointer, key, value))
            }
        }
        node_pointer = node.get_right_sibling_pointer();
    }


    (true, format!("ALL GOOD"))

}

pub fn check_tree_mapping(tree: &BPlusTreeMap<u32>, expected_keys: Vec<u32>) -> (bool, String) {
    let mut record = String::from("Wrongly mapped keys:\n");

    let mut success = true;

    for key in expected_keys {
        let p = tree.get_value(&key);
        if p.is_null() {
            record.push_str(&format!("{}\n", key));
            success = false;
        }
    }

    (success, record)
}

#[inline]
pub fn cut(length: usize) -> usize {
    if length % 2 == 0 {
        return length / 2;
    }
    else {
        return (length / 2) + 1;
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    
    #[test]
    fn test_BPlusTree_proper() {
        let mut tree: BPlusTreeMap<u32> = BPlusTreeMap::new(String::from("test"));
        let mut inserts = FnvHashSet::default();
        for _ in 0..1000 {
            let insert: u32 = rand::random_range(0..1000);
            inserts.insert(insert);
        }
        
        // let mut log = Vec::new();
        let mut inserted = Vec::new();
        for count in 0..10_000 {
            let item = pop_from_hashset(&mut inserts);
            if item.is_none() {
                break
            } else {
                let item = item.unwrap();
                println!("{},", item);
                tree.insert(&item, ptr(item as usize));
                inserted.push(item);
            }
            if rand::random_bool(0.1) {
                let delete = inserted.swap_remove(rand::random_range(0..inserted.len()));
                println!("-{},  {}", delete, count);
                tree.delete_key(&delete).unwrap();
            }
        }

        let (height_is_correct, height_error) = check_tree_height(&tree);
        let (order_is_correct, order_error) = check_tree_ordering(&tree);
        let (tree_leaves_are_correctly_paired, leaf_pair_error) = check_tree_leafpairs(&tree);
        let (tree_is_accurate, missing_keys) = check_tree_mapping(&tree, inserted);

        let mut we_should_panic = false;
        if !height_is_correct {
            println!("tree:\n{}", tree);
            println!("{}", height_error);
            we_should_panic = true;
        }

        if !order_is_correct {
            println!("tree:\n{}", tree);
            println!("{}", order_error);

            we_should_panic = true;
        }

        if !tree_leaves_are_correctly_paired {
            println!("tree:\n{}", tree);
            println!("{}", leaf_pair_error);

            we_should_panic = true;
        }

        if !tree_is_accurate {
            println!("tree:\n{}", tree);
            println!("{}", missing_keys);

            we_should_panic = true;
        }

        if we_should_panic {
            panic!()
        } else {
            println!("SUCCESS!?!?!\n{}", tree);
        }

    }


}